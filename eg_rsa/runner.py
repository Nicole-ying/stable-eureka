from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from eg_rsa.diagnostics.attribution import RewardAttributionAnalyzer
from eg_rsa.diagnostics.hack_detectors import RewardHackDetector
from eg_rsa.memory.memory_card import MemoryCard
from eg_rsa.memory.memory_store import MemoryStore
from eg_rsa.reward.operators import RewardEditOperatorApplier
from eg_rsa.reward.safe_compiler import SafeRewardCompiler
from eg_rsa.reward.schema import RewardSchema


class EGRSARunner:
    """Minimal EG-RSA runner.

    This first runner validates the core research loop without touching the
    original StableEureka execution path:
      reward schema -> compile -> diagnose trajectories -> retrieve memory ->
      apply constrained edit plan -> write new schema and memory card.

    PPO training and live trajectory recording will be connected in the next
    implementation step.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        schema = self._load_schema(Path(self.config["eg_rsa"]["initial_schema_path"]))
        trajectories = self._load_trajectories(Path(self.config["eg_rsa"]["trajectory_path"]))

        compiled_code = SafeRewardCompiler.compile(schema)
        self._write_text(self.output_dir / "compiled_reward.py", compiled_code)

        attribution = RewardAttributionAnalyzer.analyze(trajectories)
        detector = RewardHackDetector(**self.config.get("hack_detector", {}))
        diagnostics = detector.detect(trajectories, attribution)
        diagnostic_report = {
            "attribution": attribution,
            "diagnostics": diagnostics,
        }
        self._write_json(self.output_dir / "diagnostic_report.json", diagnostic_report)

        memory_store = MemoryStore(self.output_dir / "memory" / "memory_cards.jsonl")
        retrieved = memory_store.retrieve(
            diagnostics.get("failure_modes", []),
            env_family=self.config.get("environment", {}).get("family", "unknown"),
            top_k=int(self.config.get("memory", {}).get("top_k", 3)),
        )
        self._write_json(
            self.output_dir / "retrieved_memory.json",
            [card.to_dict() for card in retrieved],
        )

        edit_plan = self._load_edit_plan(Path(self.config["eg_rsa"]["edit_plan_path"]))
        new_schema = RewardEditOperatorApplier.apply(schema, edit_plan)
        self._write_json(self.output_dir / "reward_schema_next.json", new_schema.to_dict())

        memory_card = MemoryCard(
            memory_id=f"memory_{new_schema.version:04d}",
            env_family=self.config.get("environment", {}).get("family", "unknown"),
            failure_modes=diagnostics.get("failure_modes", []),
            reward_attribution=attribution,
            edit_plan=edit_plan,
            outcome={
                "note": "Outcome placeholder. Fill after training the edited reward schema.",
                "hack_score_before": diagnostics.get("hack_score", 0.0),
            },
            lesson="Initial EG-RSA memory card generated from diagnostics and constrained edit plan.",
            metadata={"config_path": str(self.config_path)},
        )
        memory_store.append(memory_card)
        self._write_json(self.output_dir / "latest_memory_card.json", memory_card.to_dict())

        print(f"EG-RSA minimal loop finished. Outputs saved to: {self.output_dir}")

    @staticmethod
    def _load_schema(path: Path) -> RewardSchema:
        with path.open("r", encoding="utf-8") as f:
            return RewardSchema.from_dict(json.load(f))

    @staticmethod
    def _load_trajectories(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
            if isinstance(data, dict) and "trajectories" in data:
                return data["trajectories"]
            if not isinstance(data, list):
                raise ValueError("Trajectory file must contain a list or {'trajectories': [...]} object")
            return data

    @staticmethod
    def _load_edit_plan(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("edit_plan", [])
        if not isinstance(data, list):
            raise ValueError("Edit plan must be a list or {'edit_plan': [...]} object")
        return data

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
