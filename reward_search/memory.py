"""
Persistent memory for the reward search. JSONL-based, simple and transparent.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class RewardSearchMemory:
    """Structured memory for the reward search process."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_path = self.memory_dir / "experiments.jsonl"
        self.observations_path = self.memory_dir / "observations.md"
        self.strategy_path = self.memory_dir / "strategy.md"

    # --- Experiments log ---
    def log_experiment(self, entry: dict[str, Any]) -> None:
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.experiments_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_all_experiments(self) -> list[dict[str, Any]]:
        if not self.experiments_path.exists():
            return []
        entries = []
        with open(self.experiments_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def get_best(self) -> dict[str, Any] | None:
        entries = self.get_all_experiments()
        if not entries:
            return None
        return max(entries, key=lambda e: e.get("hidden_return_mean", -float("inf")))

    # --- Observations (what I've learned about the env) ---
    def write_observations(self, text: str) -> None:
        with open(self.observations_path, "w") as f:
            f.write(text)

    def read_observations(self) -> str:
        if not self.observations_path.exists():
            return ""
        return self.observations_path.read_text()

    def append_observation(self, text: str) -> None:
        content = self.read_observations()
        content += f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n{text}"
        self.write_observations(content)

    # --- Strategy ---
    def write_strategy(self, text: str) -> None:
        with open(self.strategy_path, "w") as f:
            f.write(text)

    def read_strategy(self) -> str:
        if not self.strategy_path.exists():
            return ""
        return self.strategy_path.read_text()
