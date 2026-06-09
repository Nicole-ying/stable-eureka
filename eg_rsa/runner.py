from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from eg_rsa.diagnostics.attribution import RewardAttributionAnalyzer
from eg_rsa.diagnostics.hack_detectors import RewardHackDetector
from eg_rsa.diagnostics.semantic_outcome import SemanticOutcomeAnalyzer
from eg_rsa.evaluation.experiment_summary import ExperimentSummary
from eg_rsa.experiments.modes import ExperimentMode
from eg_rsa.llm.client_factory import build_llm_client
from eg_rsa.llm.edit_agent import EditAgent
from eg_rsa.llm.reflection_agent import ReflectionAgent
from eg_rsa.llm.structural_search_agent import StructuralSearchAgent
from eg_rsa.memory.lesson_store import LessonStore, build_lesson_from_memory_card
from eg_rsa.memory.memory_card import MemoryCard
from eg_rsa.memory.memory_store import MemoryStore
from eg_rsa.reward.candidate_evaluator import RewardCandidateEvaluator
from eg_rsa.reward.edit_decision_gate import EditDecisionGate
from eg_rsa.reward.edit_plan_validator import EditPlanValidator
from eg_rsa.reward.operators import RewardEditOperatorApplier
from eg_rsa.reward.outcome_acceptor import OutcomeAcceptor
from eg_rsa.reward.safe_compiler import SafeRewardCompiler
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.training.eg_rsa_trainer import EGRSATrainer
from eg_rsa.agent.action_controller import AgentActionController
from eg_rsa.tools.outcome_lesson_builder import OutcomeLessonBuilder
from eg_rsa.tools.scale_audit import ScaleAuditTool
from eg_rsa.tools.trajectory_inspector import TrajectoryInspector


class EGRSARunner:
    """EG-RSA runner using real policy training and rollout trajectories."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = ExperimentMode.from_config(self.config)
        llm_client = build_llm_client(self.config) if self.mode.use_llm_edit else None
        self.reflection_agent = ReflectionAgent(llm_client=llm_client)
        self.edit_agent = EditAgent(llm_client=llm_client)
        self.structural_search_agent = StructuralSearchAgent(llm_client=llm_client)
        self.agent_action_controller = AgentActionController(self.config.get("agent_action_controller", {}))
        self.structural_context = self._load_structural_context()

    def run(self) -> None:
        schema = self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))
        iterations = int(self.config.get("eg_rsa", {}).get("iterations", 1))
        if self.mode.one_shot:
            iterations = 1

        memory_store = MemoryStore(self.output_dir / "memory" / "memory_cards.jsonl")
        lesson_store = LessonStore(self.output_dir / "memory" / "lesson_cards.jsonl")
        run_history: List[Dict[str, Any]] = []
        best_schema = schema
        best_score = -float("inf")
        task_description = self._load_task_description()
        self._write_json(self.output_dir / "experiment_mode.json", self.mode.to_dict())
        self._write_json(self.output_dir / "structural_context.json", self.structural_context)

        pending_memory_id: Optional[str] = None
        pending_before: Optional[Dict[str, Any]] = None
        pending_card_dict: Optional[Dict[str, Any]] = None
        stop_reason: Optional[str] = None
        next_init_model_path: Optional[Path] = None

        for iteration in range(iterations):
            iter_dir = self.output_dir / f"iteration_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(iter_dir / "reward_schema.json", schema.to_dict())
            self._write_text(iter_dir / "compiled_reward.py", SafeRewardCompiler.compile(schema))

            trainer = EGRSATrainer(self.config, iter_dir)
            current_init_model_path = next_init_model_path
            next_init_model_path = None
            trajectories = trainer.train_and_record(schema, init_model_path=current_init_model_path)
            raw_attribution = RewardAttributionAnalyzer.analyze(trajectories)
            semantic_outcome = SemanticOutcomeAnalyzer.analyze(trajectories, schema, self.structural_context)
            self._write_json(iter_dir / "semantic_outcome.json", semantic_outcome)

            diagnostics = self._diagnose(trajectories, raw_attribution, semantic_outcome)
            diagnostic_report = self._diagnostic_report(raw_attribution, diagnostics, semantic_outcome)
            self._write_json(iter_dir / "diagnostic_report.json", diagnostic_report)

            trajectory_inspection = TrajectoryInspector.inspect(trajectories)
            self._write_json(iter_dir / "trajectory_inspection.json", trajectory_inspection)

            task_score = self._task_score(trajectories)
            current_metrics = self._current_metrics(task_score, diagnostics, semantic_outcome)
            current_selection_score = float(current_metrics.get("selection_score", task_score))

            rollback_applied = False
            outcome_decision_dict: Dict[str, Any] = {}

            if self.mode.use_memory and pending_memory_id and pending_before and pending_card_dict:
                transition = self._make_outcome(pending_before, current_metrics)
                outcome_decision = OutcomeAcceptor.decide(
                    pending_before,
                    current_metrics,
                    self.config.get("outcome_acceptor", {}),
                )
                outcome_decision_dict = outcome_decision.to_dict()
                transition["outcome_decision"] = outcome_decision_dict

                memory_store.update_outcome(pending_memory_id, transition)
                measured_card = dict(pending_card_dict)
                measured_card["outcome"] = transition
                measured_card.setdefault("metadata", {})["outcome_decision"] = outcome_decision_dict
                measured_card.setdefault("metadata", {})["semantic_after"] = semantic_outcome
                lesson_card = build_lesson_from_memory_card(measured_card)
                lesson_store.append(lesson_card)
                self._write_json(
                    iter_dir / "memory_transition.json",
                    {
                        "memory_id": pending_memory_id,
                        "outcome": transition,
                        "outcome_decision": outcome_decision_dict,
                        "lesson_card": lesson_card,
                    },
                )

                before_schema_for_lesson = (
                    pending_card_dict.get("metadata", {}).get("schema_snapshot", {})
                    if isinstance(pending_card_dict, dict)
                    else {}
                )
                outcome_lesson = OutcomeLessonBuilder.build(
                    before_schema=before_schema_for_lesson,
                    after_schema=schema.to_dict(),
                    edit_plan=pending_card_dict.get("edit_plan", []) if isinstance(pending_card_dict, dict) else [],
                    before_metrics=pending_before,
                    after_metrics=current_metrics,
                    outcome_decision=outcome_decision_dict,
                    attribution_after=raw_attribution,
                )
                self._write_json(iter_dir / "outcome_lesson.json", outcome_lesson)
                outcome_lesson_path = self.output_dir / "memory" / "outcome_lessons.jsonl"
                outcome_lesson_path.parent.mkdir(parents=True, exist_ok=True)
                with outcome_lesson_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(outcome_lesson, ensure_ascii=False) + "\n")

                if outcome_decision.accepted_for_best and current_selection_score > best_score:
                    best_score = current_selection_score
                    best_schema = schema
                    self._write_json(self.output_dir / "best_reward_schema.json", best_schema.to_dict())

                if outcome_decision.rollback_recommended:
                    rollback_applied = True
                    self._write_json(
                        iter_dir / "rollback_decision.json",
                        {
                            "iteration": iteration,
                            "pending_memory_id": pending_memory_id,
                            "outcome_decision": outcome_decision_dict,
                            "rollback_target": "best_accepted_schema",
                            "best_score": best_score,
                            "current_selection_score": current_selection_score,
                            "reason": outcome_decision.reason,
                            "replan_immediately": True,
                        },
                    )
                    schema = best_schema
                    self._write_json(iter_dir / "reward_schema_after_rollback.json", schema.to_dict())

                pending_memory_id = None
                pending_before = None
                pending_card_dict = None
            elif current_selection_score > best_score:
                best_score = current_selection_score
                best_schema = schema
                self._write_json(self.output_dir / "best_reward_schema.json", best_schema.to_dict())

            retrieved_dicts: List[Dict[str, Any]] = []
            retrieved_lessons: List[Dict[str, Any]] = []
            retrieved_outcome_lessons: List[Dict[str, Any]] = []
            if self.mode.use_memory:
                retrieved = memory_store.retrieve(
                    diagnostics.get("failure_modes", []),
                    env_family=self.config.get("environment", {}).get("family", "unknown"),
                    top_k=int(self.config.get("memory", {}).get("top_k", 3)),
                )
                retrieved_dicts = [card.to_dict() for card in retrieved]
                retrieved_lessons = lesson_store.retrieve(
                    diagnostics.get("failure_modes", []),
                    top_k=int(self.config.get("memory", {}).get("lesson_top_k", 5)),
                )
                retrieved_outcome_lessons = self._load_recent_jsonl(
                    self.output_dir / "memory" / "outcome_lessons.jsonl",
                    limit=int(self.config.get("memory", {}).get("outcome_lesson_top_k", 5)),
                )
                for lesson in retrieved_outcome_lessons:
                    wrapped = dict(lesson)
                    wrapped["source"] = "outcome_lesson"
                    retrieved_lessons.append(wrapped)

            self._write_json(iter_dir / "retrieved_memory.json", retrieved_dicts)
            self._write_json(iter_dir / "retrieved_lessons.json", retrieved_lessons)
            self._write_json(iter_dir / "retrieved_outcome_lessons.json", retrieved_outcome_lessons)

            agent_action_decision = self.agent_action_controller.decide(
                diagnostic_report=diagnostic_report,
                semantic_outcome=semantic_outcome,
                retrieved_lessons=retrieved_lessons,
            )
            self._write_json(iter_dir / "agent_action_decision.json", agent_action_decision.to_dict())

            should_edit = iteration < iterations - 1
            gate_result = None
            candidate_result = None
            structural_response: Dict[str, Any] = {}
            reflection_report: Dict[str, Any] = {}
            validation = EditPlanValidator.validate(schema, [], structural_context=self.structural_context)
            edit_plan: List[Dict[str, Any]] = []
            edit_decision = "final" if iteration == iterations - 1 else "edit"
            next_action = "final_iteration" if iteration == iterations - 1 else "apply_edit"

            if should_edit:
                agent_active = bool(self.config.get("agent_action_controller", {}).get("active", True))
                agent_action = agent_action_decision.normalized_action()

                if agent_active and agent_action == "continue_training":
                    reflection_report = {
                        "strategy": {
                            "recommended_next_action": "continue_training",
                            "plan_type": "continue_training",
                            "max_reasonable_edits": 0,
                        },
                        "rationale": agent_action_decision.reason_summary,
                        "source": "AgentActionController",
                    }
                    self._write_json(iter_dir / "reflection_report.json", reflection_report)
                    edit_response = {
                        "diagnosis": agent_action_decision.reason_summary,
                        "diagnostic_analysis": {
                            "edit_need": "no_edit",
                            "reason": "AgentActionController selected continue_training.",
                        },
                        "reward_editor": {
                            "edit_decision": "no_edit",
                            "next_action": "continue_training",
                        },
                        "auditor_check": {
                            "approved": True,
                            "final_action": "continue_training",
                            "issues": [],
                        },
                        "edit_plan": [],
                        "agent_action_decision": agent_action_decision.to_dict(),
                    }
                    edit_decision = "no_edit"
                    next_action = "continue_training"
                else:
                    reflection_report = self.reflection_agent.reflect(
                        task_description,
                        schema.to_dict(),
                        diagnostic_report,
                        retrieved_dicts,
                        retrieved_lessons,
                    )
                    self._write_json(iter_dir / "reflection_report.json", reflection_report)
                    edit_response = self.edit_agent.generate_edit_plan(
                        task_description,
                        schema.to_dict(),
                        diagnostic_report,
                        retrieved_dicts,
                        retrieved_lessons,
                        reflection_report,
                    )
                    edit_decision = self._extract_edit_decision(edit_response)
                    next_action = self._extract_next_action(edit_response)
                    plan_metadata = self._extract_plan_metadata(edit_response, reflection_report)
                    edit_plan, validation, candidate_result, gate_result, next_action = self._validate_evaluate_and_gate_edit_plan(
                        schema,
                        edit_response.get("edit_plan", []),
                        diagnostic_report,
                        trajectories,
                        edit_decision,
                        next_action,
                        plan_metadata,
                    )
                    if not edit_plan and next_action == "structural_search":
                        structural_response = self.structural_search_agent.generate_structural_edit(
                            task_description,
                            schema.to_dict(),
                            diagnostic_report,
                            retrieved_lessons,
                            self.structural_context,
                        )
                        self._write_json(iter_dir / "structural_search_response.json", structural_response)
                        sp_meta = self._extract_plan_metadata(structural_response, reflection_report)
                        edit_plan, validation, candidate_result, gate_result, next_action = self._validate_evaluate_and_gate_edit_plan(
                            schema,
                            structural_response.get("edit_plan", []),
                            diagnostic_report,
                            trajectories,
                            self._extract_edit_decision(structural_response),
                            self._extract_next_action(structural_response),
                            sp_meta,
                        )
                        edit_response["structural_search_response"] = structural_response
            else:
                edit_response = {"diagnosis": "Final iteration; edit generation skipped.", "edit_plan": []}

            self._write_json(iter_dir / "edit_response.json", edit_response)
            self._write_json(iter_dir / "edit_validation.json", validation.to_dict())

            if edit_plan:
                scale_audit = ScaleAuditTool.audit(
                    schema=schema,
                    edit_plan=edit_plan,
                    trajectories=trajectories,
                    config=self.config.get("scale_audit", {}),
                )
                self._write_json(iter_dir / "scale_audit.json", scale_audit)

                scale_audit_active = bool(self.config.get("scale_audit", {}).get("active", True))
                if scale_audit_active and not bool(scale_audit.get("audit_pass", True)):
                    validation.errors.append(
                        "ScaleAuditTool blocked edit_plan because it may dominate terminal incentives."
                    )
                    edit_response.setdefault("auditor_check", {})
                    edit_response["auditor_check"]["scale_audit_blocked"] = True
                    edit_response["auditor_check"]["scale_audit"] = scale_audit
                    edit_plan = []
                    next_action = "continue_training"

            if candidate_result is not None:
                self._write_json(iter_dir / "candidate_evaluation.json", candidate_result.to_dict())
            if gate_result is not None:
                self._write_json(iter_dir / "edit_gate.json", gate_result.to_dict())
            self._write_json(iter_dir / "edit_plan.json", {"edit_plan": edit_plan})

            outcome = {
                "status": "pending" if should_edit and edit_plan else "not_applicable",
                "before": current_metrics,
                "after": {},
                "delta": {},
                "note": "Pending until the edited schema is trained in the next iteration." if should_edit and edit_plan else "No edit was generated for this iteration.",
            }
            memory_card = MemoryCard(
                memory_id=f"iter_{iteration:03d}_to_{iteration + 1:03d}",
                env_family=self.config.get("environment", {}).get("family", "unknown"),
                failure_modes=diagnostics.get("failure_modes", []),
                reward_attribution=raw_attribution if self.mode.use_attribution else {},
                edit_plan=edit_plan,
                outcome=outcome,
                lesson=edit_response.get("diagnosis", "Generated by EG-RSA edit agent."),
                metadata={
                    "config_path": str(self.config_path),
                    "iteration": iteration,
                    "schema_snapshot": schema.to_dict(),
                    "validation_errors": validation.errors,
                    "validation_warnings": getattr(validation, "warnings", []),
                    "candidate_evaluation": candidate_result.to_dict() if candidate_result is not None else {},
                    "gate_result": gate_result.to_dict() if gate_result is not None else {},
                    "reflection_report": reflection_report,
                    "semantic_outcome": semantic_outcome,
                    "structural_response": structural_response,
                    "outcome_decision": outcome_decision_dict,
                    "rollback_applied": rollback_applied,
                    "experiment_mode": self.mode.to_dict(),
                    "edit_generated": should_edit,
                    "edit_decision": edit_decision,
                    "next_action": next_action,
                    "agent_analysis": {
                        "reflection_report": reflection_report,
                        "diagnostic_analysis": edit_response.get("diagnostic_analysis", {}),
                        "memory_reflection": edit_response.get("memory_reflection", {}),
                        "reward_editor": edit_response.get("reward_editor", {}),
                        "auditor_check": edit_response.get("auditor_check", {}),
                        "distilled_lessons": edit_response.get("distilled_lessons", {}),
                    },
                },
            )

            if self.mode.use_memory and should_edit and edit_plan:
                memory_store.append(memory_card)
                pending_memory_id = memory_card.memory_id
                pending_before = current_metrics
                pending_card_dict = memory_card.to_dict()
            self._write_json(iter_dir / "memory_card.json", memory_card.to_dict())

            run_history.append(
                {
                    "iteration": iteration,
                    "task_score": task_score,
                    "semantic_score": semantic_outcome.get("semantic_score", 0.0),
                    "selection_score": current_metrics.get("selection_score", current_selection_score),
                    "hack_score": diagnostics.get("hack_score", 0.0),
                    "failure_modes": diagnostics.get("failure_modes", []),
                    "dominant_component": diagnostics.get("dominant_component"),
                    "dominant_component_ratio": diagnostics.get("dominant_component_ratio", 0.0),
                    "benign_terminal_dominance": diagnostics.get("benign_terminal_dominance", False),
                    "semantic_outcome": semantic_outcome,
                    "trajectory_inspection": trajectory_inspection,
                    "agent_action_decision": agent_action_decision.to_dict(),
                    "retrieved_outcome_lessons": retrieved_outcome_lessons,
                    "reflection_strategy": reflection_report.get("strategy", {}) if reflection_report else {},
                    "edit_plan": edit_plan,
                    "edit_decision": edit_decision,
                    "next_action": next_action,
                    "structural_search_used": bool(structural_response),
                    "edit_diagnosis": edit_response.get("diagnosis", ""),
                    "edit_validation_errors": validation.errors,
                    "edit_validation_warnings": getattr(validation, "warnings", []),
                    "candidate_evaluation": candidate_result.to_dict() if candidate_result is not None else {},
                    "edit_gate": gate_result.to_dict() if gate_result is not None else {},
                    "outcome_decision": outcome_decision_dict,
                    "rollback_applied": rollback_applied,
                    "best_score": best_score,
                    "edit_generated": should_edit,
                    "experiment_mode": self.mode.to_dict(),
                }
            )
            self._write_json(self.output_dir / "run_history.json", run_history)

            if should_edit:
                if edit_plan:
                    schema = RewardEditOperatorApplier.apply(schema, edit_plan)
                    self._write_json(iter_dir / "reward_schema_next.json", schema.to_dict())
                elif next_action == "continue_training":
                    model_path = iter_dir / "model.zip"
                    if model_path.exists():
                        next_init_model_path = model_path
                        self._write_json(
                            iter_dir / "continue_training.json",
                            {
                                "next_action": next_action,
                                "continued_schema": True,
                                "init_model_path_for_next_iteration": str(model_path),
                                "reason": edit_response.get("diagnosis", ""),
                            },
                        )
                        self._write_json(iter_dir / "reward_schema_next.json", schema.to_dict())
                    else:
                        stop_reason = "Stopped after continue_training because current iteration model.zip was not found."
                        self._write_json(
                            iter_dir / "stop_reason.json",
                            {"next_action": next_action, "reason": stop_reason, "diagnosis": edit_response.get("diagnosis", "")},
                        )
                        break
                elif next_action in {"early_stop", "structural_search"}:
                    stop_reason = f"Stopped after next_action={next_action}: {edit_response.get('diagnosis', '')}"
                    self._write_json(
                        iter_dir / "stop_reason.json",
                        {"next_action": next_action, "reason": stop_reason, "diagnosis": edit_response.get("diagnosis", "")},
                    )
                    break

        self._write_json(
            self.output_dir / "best_summary.json",
            {"best_score": best_score, "best_schema": best_schema.to_dict(), "stop_reason": stop_reason},
        )
        ExperimentSummary.save(self.output_dir)
        print(f"EG-RSA multi-iteration run finished. Outputs saved to: {self.output_dir}")

    def _validate_evaluate_and_gate_edit_plan(self, schema: RewardSchema, raw_edit_plan: List[Dict[str, Any]], diagnostic_report: Dict[str, Any], trajectories: List[Dict[str, Any]], edit_decision: str, next_action: str, plan_metadata: Optional[Dict[str, Any]] = None):
        if edit_decision in {"no_edit", "need_more_evidence"} or raw_edit_plan == []:
            validation = EditPlanValidator.validate(schema, [], structural_context=self.structural_context)
            validation.errors.append(f"LLM chose {edit_decision} with next_action={next_action}; no schema edit applied.")
            return [], validation, None, None, next_action
        plan_metadata = plan_metadata or {}
        is_atomic = plan_metadata.get("atomicity") == "atomic" and plan_metadata.get("plan_type") == "coupled_rebalancing"
        validation = EditPlanValidator.validate(schema, raw_edit_plan, structural_context=self.structural_context)
        candidate_result = None
        gate_result = None
        edit_plan: List[Dict[str, Any]] = []
        if is_atomic and validation.rejected_edits:
            validation.errors.append("Atomic coupled package rejected before execution because at least one edit failed validation; partial execution is forbidden.")
            return [], validation, None, None, "structural_search"
        if validation.valid_edits:
            candidate_result = RewardCandidateEvaluator.evaluate(validation.valid_edits, trajectories, self.config.get("candidate_evaluator", {}))
            if is_atomic and len(candidate_result.accepted_edits) != len(validation.valid_edits):
                validation.errors.extend(candidate_result.warnings)
                validation.errors.append("Atomic coupled package rejected because candidate evaluation removed part of the package; partial execution is forbidden.")
                return [], validation, candidate_result, None, "structural_search"
            if not candidate_result.accepted_edits:
                validation.errors.extend(candidate_result.warnings)
                return [], validation, candidate_result, None, "structural_search"
            gate_result = EditDecisionGate.apply(schema, candidate_result.accepted_edits, diagnostic_report, self.config.get("edit_gate", {}), plan_metadata)
            edit_plan = gate_result.accepted_edits
            if is_atomic and len(edit_plan) != len(candidate_result.accepted_edits):
                validation.errors.extend(gate_result.warnings)
                validation.errors.append("Atomic coupled package rejected because gate removed part of the package; partial execution is forbidden.")
                return [], validation, candidate_result, gate_result, "structural_search"
            next_action = "apply_edit" if edit_plan else "structural_search"
            if not edit_plan:
                validation.errors.extend(gate_result.warnings)
        elif diagnostic_report.get("diagnostics", {}).get("dominant_component") and not self.mode.use_llm_edit:
            edit_plan = EditPlanValidator.safe_fallback(schema, diagnostic_report.get("diagnostics", {}))
            next_action = "apply_edit"
            validation.errors.append("No valid edit remained; used non-LLM safe fallback edit plan.")
        else:
            validation.errors.append("No valid edit remained; skipped schema edit.")
        if not self.mode.use_operator_constraints:
            validation.errors.append("Operator constraints disabled for ablation; no schema edit was applied.")
            edit_plan = []
        return edit_plan, validation, candidate_result, gate_result, next_action

    @staticmethod
    def _extract_edit_decision(edit_response: Dict[str, Any]) -> str:
        if isinstance(edit_response, dict):
            editor = edit_response.get("reward_editor", {})
            if isinstance(editor, dict) and isinstance(editor.get("edit_decision"), str):
                return editor["edit_decision"]
            diagnostic = edit_response.get("diagnostic_analysis", {})
            if isinstance(diagnostic, dict):
                edit_need = diagnostic.get("edit_need")
                if edit_need in {"no_edit", "need_more_evidence"}:
                    return str(edit_need)
                if edit_need in {"must_edit", "edit"}:
                    return "edit"
        return "edit"

    @staticmethod
    def _extract_next_action(edit_response: Dict[str, Any]) -> str:
        if isinstance(edit_response, dict):
            editor = edit_response.get("reward_editor", {})
            if isinstance(editor, dict) and isinstance(editor.get("next_action"), str):
                return editor["next_action"]
            auditor = edit_response.get("auditor_check", {})
            if isinstance(auditor, dict) and isinstance(auditor.get("final_action"), str):
                return auditor["final_action"]
            diagnostic = edit_response.get("diagnostic_analysis", {})
            if isinstance(diagnostic, dict) and diagnostic.get("edit_need") in {"no_edit", "need_more_evidence"}:
                return "continue_training"
        return "apply_edit"

    @staticmethod
    def _extract_plan_metadata(edit_response: Dict[str, Any], reflection_report: Dict[str, Any]) -> Dict[str, Any]:
        editor = edit_response.get("reward_editor", {}) if isinstance(edit_response, dict) else {}
        strategy = reflection_report.get("strategy", {}) if isinstance(reflection_report, dict) else {}
        value = edit_response.get("max_reasonable_edits") or editor.get("max_reasonable_edits") or strategy.get("max_reasonable_edits") or 1
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 1
        return {"plan_type": edit_response.get("plan_type") or editor.get("plan_type") or strategy.get("plan_type") or "single_edit", "atomicity": edit_response.get("atomicity") or editor.get("atomicity") or strategy.get("atomicity") or "separable", "max_reasonable_edits": value, "reflection_strategy": strategy}

    @staticmethod
    def _make_outcome(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        keys = ["task_score", "semantic_score", "selection_score", "hack_score", "success_episode_rate", "terminal_reward_paid_episode_rate", "stable_landing_episode_rate", "safe_contact_episode_rate"]
        delta = {key: float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0) for key in keys}
        return {"status": "measured", "before": before, "after": after, "delta": delta, "note": "Measured after training the edited schema in the next iteration."}

    def _diagnose(self, trajectories: List[dict], attribution: dict, semantic_outcome: Dict[str, Any]) -> dict:
        if not self.mode.use_hack_detector:
            return {"hack_flags": {}, "failure_modes": [], "hack_score": 0.0, "suspected_components": [], "dominant_component": attribution.get("dominant_component") if self.mode.use_attribution else None, "dominant_component_ratio": attribution.get("dominant_component_ratio", 0.0) if self.mode.use_attribution else 0.0, "repeated_event_details": {}, "semantic_notes": []}
        diagnostics = RewardHackDetector(**self.config.get("hack_detector", {})).detect(trajectories, attribution, semantic_outcome=semantic_outcome)
        if not self.mode.use_attribution:
            diagnostics = dict(diagnostics)
            diagnostics["dominant_component"] = None
            diagnostics["dominant_component_ratio"] = 0.0
            diagnostics["suspected_components"] = []
        return diagnostics

    def _diagnostic_report(self, attribution: dict, diagnostics: dict, semantic_outcome: Dict[str, Any]) -> dict:
        return {"attribution": attribution if self.mode.use_attribution else {}, "diagnostics": diagnostics, "semantic_outcome": semantic_outcome, "experiment_mode": self.mode.to_dict()}

    def _load_task_description(self) -> str:
        path = self.config.get("eg_rsa", {}).get("task_description_path")
        if not path:
            return self.config.get("environment", {}).get("name", "")
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _load_structural_context(self) -> Dict[str, Any]:
        path = Path(self.config.get("eg_rsa", {}).get("diagnostic_spec_path", ""))
        if not path.exists():
            return {"available_events": [], "available_task_metrics": [], "preferred_success_events": []}
        spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        events = spec.get("events", {}) or {}
        metrics = spec.get("task_metrics", {}) or {}
        preferred_success_events = [cfg.get("event") for cfg in metrics.values() if isinstance(cfg, dict) and cfg.get("type") == "event_success" and cfg.get("event")]
        return {"available_events": sorted(events.keys()), "event_specs": events, "available_task_metrics": sorted(metrics.keys()), "task_metric_specs": metrics, "preferred_success_events": preferred_success_events}

    @staticmethod
    def _task_score(trajectories: List[dict]) -> float:
        if not trajectories:
            return -float("inf")
        successes = [float(t.get("summary", {}).get("success", 0.0)) for t in trajectories]
        progresses = [float(t.get("summary", {}).get("progress_score", 0.0)) for t in trajectories]
        lengths = [float(t.get("summary", {}).get("episode_length", 0.0)) for t in trajectories]
        return float(2.0 * (sum(successes) / max(1, len(successes))) + (sum(progresses) / max(1, len(progresses))) - 0.0001 * (sum(lengths) / max(1, len(lengths))))

    @staticmethod
    def _current_metrics(task_score: float, diagnostics: Dict[str, Any], semantic_outcome: Dict[str, Any]) -> Dict[str, Any]:
        flags = diagnostics.get("hack_flags", {}) or {}
        semantic_score = float(semantic_outcome.get("semantic_score", 0.0) or 0.0)
        hack_score = float(diagnostics.get("hack_score", 0.0) or 0.0)
        true_hack = bool(semantic_outcome.get("reward_repetition_risk", False) or flags.get("high_reward_low_progress", False) or flags.get("shaping_goal_mismatch", False))
        selection_score = float(task_score + semantic_score - (hack_score if true_hack else 0.0))
        return {"task_score": float(task_score), "semantic_score": semantic_score, "selection_score": selection_score, "hack_score": hack_score, "true_hack_risk": true_hack, "terminal_goal_evidence": bool(semantic_outcome.get("terminal_goal_evidence", False)), "reward_repetition_risk": bool(semantic_outcome.get("reward_repetition_risk", False)), "high_reward_low_progress": bool(flags.get("high_reward_low_progress", False)), "shaping_goal_mismatch": bool(flags.get("shaping_goal_mismatch", False)), "unstable_contact_behavior": bool(semantic_outcome.get("unstable_contact_behavior", False)), "benign_terminal_dominance": bool(diagnostics.get("benign_terminal_dominance", False)), "success_episode_rate": float(semantic_outcome.get("success_episode_rate", 0.0) or 0.0), "terminal_reward_paid_episode_rate": float(semantic_outcome.get("terminal_reward_paid_episode_rate", 0.0) or 0.0), "safe_contact_episode_rate": float(semantic_outcome.get("safe_contact_episode_rate", 0.0) or 0.0), "stable_landing_episode_rate": float(semantic_outcome.get("stable_landing_episode_rate", 0.0) or 0.0)}

    @staticmethod
    def _load_recent_jsonl(path: Path, limit: int = 5) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-max(0, int(limit)):]

    @staticmethod
    def _load_schema(path: Path) -> RewardSchema:
        with path.open("r", encoding="utf-8") as f:
            return RewardSchema.from_dict(json.load(f))

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
