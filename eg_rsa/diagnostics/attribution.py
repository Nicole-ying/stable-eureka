from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


class RewardAttributionAnalyzer:
    """Compute reward-component attribution from recorded trajectories.

    Important: the aggregate field named ``reward`` is not treated as an editable
    reward component.  It is the total generated reward and would otherwise be
    double-counted, causing the system to select ``reward`` as the dominant
    component even though no such editable component exists in the schema.
    """

    AGGREGATE_KEYS = {"reward", "total_reward", "oracle_reward", "oracle_reward_posthoc"}

    @staticmethod
    def analyze(trajectories: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        component_values: Dict[str, List[float]] = {}
        episode_rewards: List[float] = []

        for episode in trajectories:
            ep_reward = 0.0
            for step in episode.get("steps", []):
                ep_reward += float(step.get("reward", 0.0))
                for name, value in step.get("components", {}).items():
                    if name in RewardAttributionAnalyzer.AGGREGATE_KEYS:
                        continue
                    component_values.setdefault(name, []).append(float(value))
            episode_rewards.append(ep_reward)

        abs_total = sum(abs(v) for values in component_values.values() for v in values)
        component_stats: Dict[str, Dict[str, float]] = {}
        dominant_component = None
        dominant_ratio = 0.0

        for name, values in component_values.items():
            arr = np.asarray(values, dtype=float)
            component_abs_sum = float(np.sum(np.abs(arr)))
            ratio = float(component_abs_sum / abs_total) if abs_total > 1e-8 else 0.0
            trigger_rate = float(np.mean(np.abs(arr) > 1e-8)) if arr.size else 0.0
            stats = {
                "sum": float(np.sum(arr)),
                "abs_sum": component_abs_sum,
                "mean": float(np.mean(arr)) if arr.size else 0.0,
                "std": float(np.std(arr)) if arr.size else 0.0,
                "min": float(np.min(arr)) if arr.size else 0.0,
                "max": float(np.max(arr)) if arr.size else 0.0,
                "ratio": ratio,
                "trigger_rate": trigger_rate,
            }
            component_stats[name] = stats
            if ratio > dominant_ratio:
                dominant_ratio = ratio
                dominant_component = name

        return {
            "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episode_reward_std": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "component_stats": component_stats,
            "dominant_component": dominant_component,
            "dominant_component_ratio": dominant_ratio,
        }
