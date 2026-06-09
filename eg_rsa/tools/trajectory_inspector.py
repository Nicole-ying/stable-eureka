from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TrajectoryInspector:
    """Summarize recorded trajectories for agent-readable evidence.

    This tool is read-only. It does not change training, reward search, or
    acceptance logic. It turns rollout summaries into compact evidence that
    the agent can use before deciding the next action.
    """

    name = "trajectory_inspector"

    @staticmethod
    def load(path: str | Path) -> List[Dict[str, Any]]:
        path = Path(path)
        if path.suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("trajectories", [])
        return []

    @staticmethod
    def inspect(trajectories: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        rows = []
        for idx, traj in enumerate(trajectories or []):
            summary = traj.get("summary", {}) or {}

            toggle_count = TrajectoryInspector._first_number(
                summary,
                [
                    "contact_toggle_count",
                    "max_event_toggle_count",
                    "event_toggle_count",
                    "repeated_event_count",
                ],
                default=0.0,
            )

            rows.append(
                {
                    "episode": idx,
                    "success": float(summary.get("success", 0.0) or 0.0),
                    "progress_score": float(summary.get("progress_score", 0.0) or 0.0),
                    "episode_length": int(summary.get("episode_length", 0) or 0),
                    "safe_contact_rate": float(summary.get("safe_contact_rate", 0.0) or 0.0),
                    "stable_landing_rate": float(summary.get("stable_landing_rate", 0.0) or 0.0),
                    "contact_rate": float(summary.get("contact_rate", 0.0) or 0.0),
                    "both_contact_rate": float(summary.get("both_contact_rate", 0.0) or 0.0),
                    "contact_toggle_count": int(toggle_count),
                    "episode_reward": float(summary.get("episode_reward", 0.0) or 0.0),
                    "oracle_reward_posthoc": float(summary.get("oracle_reward_posthoc", 0.0) or 0.0),
                }
            )

        successes = [r for r in rows if r["success"] > 0.0]
        failures = [r for r in rows if r["success"] <= 0.0]
        failures.sort(key=lambda r: (r["progress_score"], -r["episode_length"]))
        successes.sort(key=lambda r: r["episode_length"])

        return {
            "tool": TrajectoryInspector.name,
            "num_episodes": len(rows),
            "success_count": len(successes),
            "success_rate": len(successes) / max(1, len(rows)),
            "episode_length_mean": TrajectoryInspector._mean(rows, "episode_length"),
            "safe_contact_rate_mean": TrajectoryInspector._mean(rows, "safe_contact_rate"),
            "stable_landing_rate_mean": TrajectoryInspector._mean(rows, "stable_landing_rate"),
            "contact_toggle_mean": TrajectoryInspector._mean(rows, "contact_toggle_count"),
            "contact_toggle_max": max((r["contact_toggle_count"] for r in rows), default=0),
            "worst_episodes": failures[:top_k],
            "best_episodes": successes[:top_k],
        }

    @staticmethod
    def inspect_file(path: str | Path, top_k: int = 3) -> Dict[str, Any]:
        return TrajectoryInspector.inspect(TrajectoryInspector.load(path), top_k=top_k)

    @staticmethod
    def _mean(rows: List[Dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return float(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows))

    @staticmethod
    def _first_number(summary: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in summary:
                try:
                    return float(summary.get(key, default) or default)
                except (TypeError, ValueError):
                    continue
        return float(default)
