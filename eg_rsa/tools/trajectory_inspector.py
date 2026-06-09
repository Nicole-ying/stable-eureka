from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TrajectoryInspector:
    """Summarize recorded trajectories for agent-readable evidence."""

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
        return data if isinstance(data, list) else data.get("trajectories", [])

    @staticmethod
    def inspect(trajectories: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        rows = []
        for idx, traj in enumerate(trajectories or []):
            summary = traj.get("summary", {}) or {}
            rows.append({
                "episode": idx,
                "success": float(summary.get("success", 0.0) or 0.0),
                "progress_score": float(summary.get("progress_score", 0.0) or 0.0),
                "episode_length": int(summary.get("episode_length", 0) or 0),
                "safe_contact_rate": float(summary.get("safe_contact_rate", 0.0) or 0.0),
                "stable_landing_rate": float(summary.get("stable_landing_rate", 0.0) or 0.0),
                "contact_toggle_count": int(summary.get("contact_toggle_count", 0) or 0),
            })

        successes = [r for r in rows if r["success"] > 0.0]
        failures = [r for r in rows if r["success"] <= 0.0]
        failures.sort(key=lambda r: (r["progress_score"], -r["episode_length"]))

        return {
            "tool": TrajectoryInspector.name,
            "num_episodes": len(rows),
            "success_count": len(successes),
            "success_rate": len(successes) / max(1, len(rows)),
            "episode_length_mean": sum(r["episode_length"] for r in rows) / max(1, len(rows)),
            "contact_toggle_mean": sum(r["contact_toggle_count"] for r in rows) / max(1, len(rows)),
            "worst_episodes": failures[:top_k],
            "best_episodes": sorted(successes, key=lambda r: r["episode_length"])[:top_k],
        }

    @staticmethod
    def inspect_file(path: str | Path, top_k: int = 3) -> Dict[str, Any]:
        return TrajectoryInspector.inspect(TrajectoryInspector.load(path), top_k=top_k)
