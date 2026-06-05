from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from eg_rsa.diagnostics.attribution import RewardAttributionAnalyzer
from eg_rsa.diagnostics.hack_detectors import RewardHackDetector
from eg_rsa.evaluation.experiment_summary import ExperimentSummary
from eg_rsa.experiments.modes import ExperimentMode
from eg_rsa.llm.client_factory import build_llm_client
from eg_rsa.llm.edit_agent import EditAgent
from eg_rsa.llm.structural_search_agent import StructuralSearchAgent
from eg_rsa.memory.lesson_store import LessonStore, build_lesson_from_memory_card
from eg_rsa.memory.memory_card import MemoryCard
from eg_rsa.memory.memory_store import MemoryStore
from eg_rsa.reward.edit_decision_gate import EditDecisionGate
from eg_rsa.reward.edit_plan_validator import EditPlanValidator
from eg_rsa.reward.operators import RewardEditOperatorApplier
from eg_rsa.reward.safe_compiler import SafeRewardCompiler
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.training.eg_rsa_trainer import EGRSATrainer


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
        self.edit_agent = EditAgent(llm_client=llm_client)
        self.structural_search_agent = StructuralSearchAgent(llm_client=llm_client)
        self.structural_context = self._load_structural_context()

    def run(self) -> None:
        schema = self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))
        iterations = int(self.config.get("eg_rsa", {}).get("iterations", 1))
        if self.mode.one_shot:
            iterations = 1

        memory_store = MemoryStore(self.output_dir / "memory" / "memory_cards.jsonl")
        lesson_store = LessonStore(self.output_dir / "memory" / "lesson_cards.jsonl")
        run_history = []
        best_schema = schema
        best_score = -float("inf")
        task_description = self._load_task_description()
        self._write_json(self.output_dir / "experiment_mode.json", self.mode.to_dict())
        self._write_json(self.output_dir / "structural_context.json", self.structural_context)

        pending_memory_id: Optional[str] = None
        pending_before: Optional[Dict[str, float]] = None
        pending_card_dict: Optional[Dict[str, Any]] = None
        stop_reason: Optional[str] = None

        for iteration in range(iterations):
            iter_dir = self.output_dir / f"iteration_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(iter_dir / "reward_schema.json", schema.to_dict())
            self._write_text(iter_dir / "compiled_reward.py", SafeRewardCompiler.compile(schema))

            trainer = EGRSATrainer(self.config, iter_dir)
            trajectories = trainer.train_and_record(schema)

            raw_attribution = RewardAttributionAnalyzer.analyze(trajectories)
            diagnostics = self._diagnose(trajectories, raw_attribution)
            diagnostic_report = self._diagnostic_report(raw_attribution, diagnostics)
            self._write_json(iter_dir / "diagnostic_report.json", diagnostic_report)

            task_score = self._task_score(trajectories, diagnostics)
            current_metrics = {"task_score": float(task_score), "hack_score": float(diagnostics.get("hack_score", 0.0))}

            if self.mode.use_memory and pending_memory_id and pending_before and pending_card_dict:
                transition = self._make_outcome(before=pending_before, after=current_metrics)
                memory_store.update_outcome(pending_memory_id, transition)
                measured_card = dict(pending_card_dict)
                measured_card["outcome"] = transition
                lesson_card = build_lesson_from_memory_card(measured_card)
                lesson_store.append(lesson_card)
                self._write_json(iter_dir / "memory_transition.json", {"memory_id": pending_memory_id, "outcome": transition, "lesson_card": lesson_card})
                pending_memory_id = None
                pending_before = None
                pending_card_dict = None

            if task_score > best_score:
                best_score = task_score
                best_schema = schema
                self._write_json(self.output_dir / "best_reward_schema.json", best_schema.to_dict())

            retrieved_dicts: List[Dict[str, Any]] = []
            retrieved_lessons: List[Dict[str, Any]] = []
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
            self._write_json(iter_dir / "retrieved_memory.json", retrieved_dicts)
            self._write_json(iter_dir / "retrieved_lessons.json", retrieved_lessons)

            should_edit = iteration < iterations - 1
            next_action = "final_iteration"
            gate_result = None
            structural_response: Dict[str, Any] = {}
            if should_edit:
                edit_response = self.edit_agent.generate_edit_plan(
                    task_description=task_description,
                    current_reward_schema=schema.to_dict(),
                    diagnostic_report=diagnostic_report,
                    retrieved_memories=retrieved_dicts,
                    retrieved_lessons=retrieved_lessons,
                )
                raw_edit_plan = edit_response.get("edit_plan", [])
                edit_decision = self._extract_edit_decision(edit_response)
                next_action = self._extract_next_action(edit_response)
                edit_plan, validation, gate_result, next_action = self._validate_and_gate_edit_plan(
                    schema=schema,
                    raw_edit_plan=raw_edit_plan,
                    diagnostic_report=diagnostic_report,
                    edit_decision=edit_decision,
                    next_action=next_action,
                )

                if not edit_plan and next_action == "structural_search":
                    structural_response = self.structural_search_agent.generate_structural_edit(
                        task_description=task_description,
                        current_reward_schema=schema.to_dict(),
                        diagnostic_report=diagnostic_report,
                        retrieved_lessons=retrieved_lessons,
                        structural_context=self.structural_context,
                    )
                    self._write_json(iter_dir / "structural_search_response.json", structural_response)
                    structural_decision = self._extract_edit_decision(structural_response)
                    structural_next_action = self._extract_next_action(structural_response)
                    edit_plan, validation, gate_result, next_action = self._validate_and_gate_edit_plan(
                        schema=schema,
                        raw_edit_plan=structural_response.get("edit_plan", []),
                        diagnostic_report=diagnostic_report,
                        edit_decision=structural_decision,
                        next_action=structural_next_action,
                    )
                    edit_response["structural_search_response"] = structural_response
            else:
                edit_response = {"diagnosis": "Final iteration; edit generation skipped.", "edit_plan": []}
                edit_decision = "final"
                edit_plan = []
                validation = EditPlanValidator.validate(schema, [], structural_context=self.structural_context)

            self._write_json(iter_dir / "edit_response.json", edit_response)
            self._write_json(iter_dir / "edit_validation.json", validation.to_dict())
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
                    "validation_errors": validation.errors,
                    "gate_result": gate_result.to_dict() if gate_result is not None else {},
                    "structural_response": structural_response,
                    "experiment_mode": self.mode.to_dict(),
                    "edit_generated": should_edit,
                    "edit_decision": edit_decision,
                    "next_action": next_action,
                    "agent_analysis": {
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

            run_history.append({
                "iteration": iteration,
                "task_score": task_score,
                "hack_score": diagnostics.get("hack_score", 0.0),
                "failure_modes": diagnostics.get("failure_modes", []),
                "dominant_component": diagnostics.get("dominant_component"),
                "dominant_component_ratio": diagnostics.get("dominant_component_ratio", 0.0),
                "edit_plan": edit_plan,
                "edit_decision": edit_decision,
                "next_action": next_action,
                "structural_search_used": bool(structural_response),
                "edit_diagnosis": edit_response.get("diagnosis", ""),
                "edit_validation_errors": validation.errors,
                "edit_gate": gate_result.to_dict() if gate_result is not None else {},
                "edit_generated": should_edit,
                "experiment_mode": self.mode.to_dict(),
            })
            self._write_json(self.output_dir / "run_history.json", run_history)

            if should_edit:
                if edit_plan:
                    schema = RewardEditOperatorApplier.apply(schema, edit_plan)
                    self._write_json(iter_dir / "reward_schema_next.json", schema.to_dict())
                elif next_action in {"early_stop", "structural_search"}:
                    stop_reason = f"Stopped after next_action={next_action}: {edit_response.get('diagnosis', '')}"
                    self._write_json(iter_dir / "stop_reason.json", {"next_action": next_action, "reason": stop_reason, "diagnosis": edit_response.get("diagnosis", "")})
                    break

        self._write_json(self.output_dir / "best_summary.json", {"best_score": best_score, "best_schema": best_schema.to_dict(), "stop_reason": stop_reason})
        ExperimentSummary.save(self.output_dir)
        print(f"EG-RSA multi-iteration run finished. Outputs saved to: {self.output_dir}")

    def _validate_and_gate_edit_plan(
        self,
        schema: RewardSchema,
        raw_edit_plan: List[Dict[str, Any]],
        diagnostic_report: Dict[str, Any],
        edit_decision: str,
        next_action: str,
    ):
        if edit_decision in {"no_edit", "need_more_evidence"} or raw_edit_plan == []:
            edit_plan: List[Dict[str, Any]] = []
            validation = EditPlanValidator.validate(schema, [], structural_context=self.structural_context)
            validation.errors.append(f"LLM chose {edit_decision} with next_action={next_action}; no schema edit applied.")
            return edit_plan, validation, None, next_action

        validation = EditPlanValidator.validate(schema, raw_edit_plan, structural_context=self.structural_context)
        gate_result = None
        if validation.valid_edits:
            gate_result = EditDecisionGate.apply(
                schema=schema,
                edit_plan=validation.valid_edits,
                diagnostic_report=diagnostic_report,
                gate_config=self.config.get("edit_gate", {}),
            )
            edit_plan = gate_result.accepted_edits
            if edit_plan:
                next_action = "apply_edit"
            else:
                validation.errors.extend(gate_result.warnings)
                next_action = "structural_search"
        elif diagnostic_report.get("diagnostics", {}).get("dominant_component") and not self.mode.use_llm_edit:
            edit_plan = EditPlanValidator.safe_fallback(schema, diagnostic_report.get("diagnostics", {}))
            next_action = "apply_edit"
            validation.errors.append("No valid edit remained; used non-LLM safe fallback edit plan.")
        else:
            edit_plan = []
            validation.errors.append("No valid edit remained; skipped schema edit.")

        if not self.mode.use_operator_constraints:
            validation.errors.append("Operator constraints disabled for ablation; no schema edit was applied.")
            edit_plan = []
        return edit_plan, validation, gate_result, next_action

    @staticmethod
    def _extract_edit_decision(edit_response: Dict[str, Any]) -> str:
        editor = edit_response.get("reward_editor", {}) if isinstance(edit_response, dict) else {}
        if isinstance(editor, dict):
            decision = editor.get("edit_decision")
            if isinstance(decision, str) and decision:
                return decision
        return "edit"

    @staticmethod
    def _extract_next_action(edit_response: Dict[str, Any]) -> str:
        editor = edit_response.get("reward_editor", {}) if isinstance(edit_response, dict) else {}
        if isinstance(editor, dict):
            action = editor.get("next_action")
            if isinstance(action, str) and action:
                return action
        auditor = edit_response.get("auditor_check", {}) if isinstance(edit_response, dict) else {}
        if isinstance(auditor, dict):
            action = auditor.get("final_action")
            if isinstance(action, str) and action:
                return action
        return "apply_edit"

    @staticmethod
    def _make_outcome(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Any]:
        return {
            "status": "measured",
            "before": before,
            "after": after,
            "delta": {
                "task_score": float(after.get("task_score", 0.0) - before.get("task_score", 0.0)),
                "hack_score": float(after.get("hack_score", 0.0) - before.get("hack_score", 0.0)),
            },
            "note": "Measured after training the edited schema in the next iteration.",
        }

    def _diagnose(self, trajectories: List[dict], attribution: dict) -> dict:
        if not self.mode.use_hack_detector:
            return {"hack_flags": {}, "failure_modes": [], "hack_score": 0.0, "suspected_components": [], "dominant_component": attribution.get("dominant_component") if self.mode.use_attribution else None, "dominant_component_ratio": attribution.get("dominant_component_ratio", 0.0) if self.mode.use_attribution else 0.0, "repeated_event_details": {}}
        detector = RewardHackDetector(**self.config.get("hack_detector", {}))
        diagnostics = detector.detect(trajectories, attribution)
        if not self.mode.use_attribution:
            diagnostics = dict(diagnostics)
            diagnostics["dominant_component"] = None
            diagnostics["dominant_component_ratio"] = 0.0
            diagnostics["suspected_components"] = []
        return diagnostics

    def _diagnostic_report(self, attribution: dict, diagnostics: dict) -> dict:
        return {"attribution": attribution if self.mode.use_attribution else {}, "diagnostics": diagnostics, "experiment_mode": self.mode.to_dict()}

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
        return {
            "available_events": sorted(events.keys()),
            "event_specs": events,
            "available_task_metrics": sorted(metrics.keys()),
            "task_metric_specs": metrics,
            "preferred_success_events": preferred_success_events,
        }

    @staticmethod
    def _task_score(trajectories: List[dict], diagnostics: dict) -> float:
        if not trajectories:
            return -float("inf")
        successes = [float(t.get("summary", {}).get("success", 0.0)) for t in trajectories]
        progresses = [float(t.get("summary", {}).get("progress_score", 0.0)) for t in trajectories]
        lengths = [float(t.get("summary", {}).get("episode_length", 0.0)) for t in trajectories]
        success_mean = sum(successes) / max(1, len(successes))
        progress_mean = sum(progresses) / max(1, len(progresses))
        length_penalty = 0.0001 * (sum(lengths) / max(1, len(lengths)))
        hack_penalty = float(diagnostics.get("hack_score", 0.0))
        return float(success_mean * 2.0 + progress_mean - hack_penalty - length_penalty)

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
