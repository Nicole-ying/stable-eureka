from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import yaml

from eg_rsa.diagnostics.attribution import RewardAttributionAnalyzer
from eg_rsa.diagnostics.hack_detectors import RewardHackDetector
from eg_rsa.evaluation.experiment_summary import ExperimentSummary
from eg_rsa.experiments.modes import ExperimentMode
from eg_rsa.llm.client_factory import build_llm_client
from eg_rsa.llm.edit_agent import EditAgent
from eg_rsa.memory.memory_card import MemoryCard
from eg_rsa.memory.memory_store import MemoryStore
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

    def run(self) -> None:
        schema = self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))
        iterations = int(self.config.get("eg_rsa", {}).get("iterations", 1))
        if self.mode.one_shot:
            iterations = 1
        memory_store = MemoryStore(self.output_dir / "memory" / "memory_cards.jsonl")
        run_history = []
        best_schema = schema
        best_score = -float("inf")
        task_description = self._load_task_description()
        self._write_json(self.output_dir / "experiment_mode.json", self.mode.to_dict())

        for iteration in range(iterations):
            iter_dir = self.output_dir / f"iteration_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(iter_dir / "reward_schema.json", schema.to_dict())
            self._write_text(iter_dir / "compiled_reward.py", SafeRewardCompiler.compile(schema))

            trainer = EGRSATrainer(self.config, iter_dir)
            trajectories = trainer.train_and_record(schema)

            attribution = RewardAttributionAnalyzer.analyze(trajectories)
            diagnostics = self._diagnose(trajectories, attribution)
            diagnostic_report = self._diagnostic_report(attribution, diagnostics)
            self._write_json(iter_dir / "diagnostic_report.json", diagnostic_report)

            task_score = self._task_score(trajectories, diagnostics)
            if task_score > best_score:
                best_score = task_score
                best_schema = schema
                self._write_json(self.output_dir / "best_reward_schema.json", best_schema.to_dict())

            retrieved_dicts = []
            if self.mode.use_memory:
                retrieved = memory_store.retrieve(
                    diagnostics.get("failure_modes", []),
                    env_family=self.config.get("environment", {}).get("family", "unknown"),
                    top_k=int(self.config.get("memory", {}).get("top_k", 3)),
                )
                retrieved_dicts = [card.to_dict() for card in retrieved]
            self._write_json(iter_dir / "retrieved_memory.json", retrieved_dicts)

            edit_response = self.edit_agent.generate_edit_plan(
                task_description=task_description,
                current_reward_schema=schema.to_dict(),
                diagnostic_report=diagnostic_report,
                retrieved_memories=retrieved_dicts,
            )
            raw_edit_plan = edit_response.get("edit_plan", [])
            validation = EditPlanValidator.validate(schema, raw_edit_plan)
            if validation.valid_edits:
                edit_plan = validation.valid_edits
            else:
                edit_plan = EditPlanValidator.safe_fallback(schema, diagnostics)
                validation.errors.append("No valid edit remained; used safe fallback edit plan.")
            if not self.mode.use_operator_constraints:
                # The runner still cannot safely execute arbitrary code rewrites.
                # We record the free-rewrite response but do not apply unsafe code.
                validation.errors.append("Operator constraints disabled for ablation; arbitrary code execution is intentionally not supported by this runner.")
                edit_plan = []
            self._write_json(iter_dir / "edit_response.json", edit_response)
            self._write_json(iter_dir / "edit_validation.json", validation.to_dict())
            self._write_json(iter_dir / "edit_plan.json", {"edit_plan": edit_plan})

            memory_card = MemoryCard(
                memory_id=f"iter_{iteration:03d}_to_{iteration + 1:03d}",
                env_family=self.config.get("environment", {}).get("family", "unknown"),
                failure_modes=diagnostics.get("failure_modes", []),
                reward_attribution=attribution if self.mode.use_attribution else {},
                edit_plan=edit_plan,
                outcome={
                    "note": "Outcome is measured after training the edited schema in the next iteration.",
                    "hack_score_before": diagnostics.get("hack_score", 0.0),
                    "task_score_before": task_score,
                },
                lesson=edit_response.get("diagnosis", "Generated by EG-RSA edit agent."),
                metadata={
                    "config_path": str(self.config_path),
                    "iteration": iteration,
                    "validation_errors": validation.errors,
                    "experiment_mode": self.mode.to_dict(),
                },
            )
            if self.mode.use_memory:
                memory_store.append(memory_card)
            self._write_json(iter_dir / "memory_card.json", memory_card.to_dict())

            run_history.append(
                {
                    "iteration": iteration,
                    "task_score": task_score,
                    "hack_score": diagnostics.get("hack_score", 0.0),
                    "failure_modes": diagnostics.get("failure_modes", []),
                    "dominant_component": diagnostics.get("dominant_component"),
                    "dominant_component_ratio": diagnostics.get("dominant_component_ratio", 0.0),
                    "edit_plan": edit_plan,
                    "edit_diagnosis": edit_response.get("diagnosis", ""),
                    "edit_validation_errors": validation.errors,
                    "experiment_mode": self.mode.to_dict(),
                }
            )
            self._write_json(self.output_dir / "run_history.json", run_history)

            if iteration < iterations - 1:
                if not edit_plan:
                    break
                schema = RewardEditOperatorApplier.apply(schema, edit_plan)
                self._write_json(iter_dir / "reward_schema_next.json", schema.to_dict())

        self._write_json(self.output_dir / "best_summary.json", {"best_score": best_score, "best_schema": best_schema.to_dict()})
        ExperimentSummary.save(self.output_dir)
        print(f"EG-RSA multi-iteration run finished. Outputs saved to: {self.output_dir}")

    def _diagnose(self, trajectories: List[dict], attribution: dict) -> dict:
        if not self.mode.use_hack_detector:
            return {
                "hack_flags": {},
                "failure_modes": [],
                "hack_score": 0.0,
                "suspected_components": [],
                "dominant_component": attribution.get("dominant_component") if self.mode.use_attribution else None,
                "dominant_component_ratio": attribution.get("dominant_component_ratio", 0.0) if self.mode.use_attribution else 0.0,
                "repeated_event_details": {},
            }
        detector = RewardHackDetector(**self.config.get("hack_detector", {}))
        return detector.detect(trajectories, attribution)

    def _diagnostic_report(self, attribution: dict, diagnostics: dict) -> dict:
        return {
            "attribution": attribution if self.mode.use_attribution else {},
            "diagnostics": diagnostics,
            "experiment_mode": self.mode.to_dict(),
        }

    def _load_task_description(self) -> str:
        path = self.config.get("eg_rsa", {}).get("task_description_path")
        if not path:
            return self.config.get("environment", {}).get("name", "")
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else ""

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
