from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eg_rsa.diagnostics.event_evaluator import EventEvaluator
from eg_rsa.diagnostics.task_metrics import TaskMetricEvaluator


class TrajectoryRecorder:
    """Record step-level rollout trajectories for EG-RSA diagnostics."""

    def __init__(
        self,
        obs_adapter,
        task_metric_evaluator: TaskMetricEvaluator,
        event_evaluator: EventEvaluator,
    ):
        self.obs_adapter = obs_adapter
        self.task_metric_evaluator = task_metric_evaluator
        self.event_evaluator = event_evaluator

    def record_policy(
        self,
        model,
        env,
        n_episodes: int,
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        trajectories: List[Dict[str, Any]] = []
        for episode_id in range(n_episodes):
            reset_out = env.reset(seed=seed + episode_id) if seed is not None else env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            steps = []
            total_reward = 0.0
            total_oracle_reward = 0.0
            t = 0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                step_out = env.step(action)
                if len(step_out) == 5:
                    next_obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    next_obs, reward, done, info = step_out
                    done = bool(done)

                reward_float = self._to_float(reward)
                info_dict = self._first_info(info)
                obs_map = info_dict.get("obs_map") or self.obs_adapter.obs_to_map(next_obs)
                events = info_dict.get("events") or self.event_evaluator.evaluate(obs_map, action)
                task_metrics = info_dict.get("task_metrics") or self.task_metric_evaluator.evaluate(obs_map, action, events)
                components = self._extract_components(info_dict)
                oracle_reward = self._to_float(info_dict.get("oracle_reward_posthoc", 0.0))

                steps.append(
                    {
                        "t": t,
                        "obs": self._to_list(obs),
                        "next_obs": self._to_list(next_obs),
                        "action": self._to_list(action),
                        "reward": reward_float,
                        "oracle_reward_posthoc": oracle_reward,
                        "components": components,
                        "task_metrics": self._float_dict(task_metrics),
                        "events": self._bool_dict(events),
                        "done": done,
                    }
                )
                total_reward += reward_float
                total_oracle_reward += oracle_reward
                obs = next_obs
                t += 1

            trajectories.append(
                {
                    "episode_id": episode_id,
                    "steps": steps,
                    "summary": self._summary(total_reward, total_oracle_reward, steps),
                }
            )
        return trajectories

    @staticmethod
    def save_jsonl(path: Path, trajectories: List[Dict[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_components(info: Dict[str, Any]) -> Dict[str, float]:
        components = info.get("components") or info.get("individual_reward") or info.get("individual_rewards") or {}
        if not isinstance(components, dict):
            return {}
        return TrajectoryRecorder._float_dict(components)

    @staticmethod
    def _summary(total_reward: float, total_oracle_reward: float, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not steps:
            return {
                "episode_reward": total_reward,
                "oracle_reward_posthoc": total_oracle_reward,
                "episode_length": 0,
                "progress_score": 0.0,
                "success": 0.0,
            }
        last_metrics = steps[-1].get("task_metrics", {})
        success = max(float(step.get("task_metrics", {}).get("success", 0.0)) for step in steps)
        repeated_event_count = TrajectoryRecorder._max_event_toggle_count(steps)
        progress_score = TrajectoryRecorder._task_progress_score(last_metrics)
        landing_quality_values = [float(step.get("task_metrics", {}).get("landing_quality", 0.0)) for step in steps]
        approach_values = [float(step.get("task_metrics", {}).get("approach_region_score", 0.0)) for step in steps]
        stability_values = [float(step.get("task_metrics", {}).get("stability", 0.0)) for step in steps]
        event_rates = TrajectoryRecorder._event_rates(steps)
        return {
            "episode_reward": float(total_reward),
            "oracle_reward_posthoc": float(total_oracle_reward),
            "episode_length": len(steps),
            "progress_score": float(progress_score),
            "success": float(success),
            "landing_quality_final": float(last_metrics.get("landing_quality", 0.0)),
            "landing_quality_max": float(max(landing_quality_values) if landing_quality_values else 0.0),
            "approach_region_final": float(last_metrics.get("approach_region_score", 0.0)),
            "approach_region_max": float(max(approach_values) if approach_values else 0.0),
            "stability_final": float(last_metrics.get("stability", 0.0)),
            "stability_mean": float(np.mean(stability_values) if stability_values else 0.0),
            "contact_rate": float(event_rates.get("contact", 0.0)),
            "both_contact_rate": float(event_rates.get("both_contact", 0.0)),
            "safe_contact_rate": float(event_rates.get("safe_contact", 0.0)),
            "stable_landing_rate": float(event_rates.get("stable_landing_condition", 0.0)),
            "max_event_toggle_count": int(repeated_event_count),
        }

    @staticmethod
    def _task_progress_score(metrics: Dict[str, Any]) -> float:
        # Prefer aligned composite task proxies, then fall back to older names.
        for key in [
            "landing_quality",
            "approach_and_stability",
            "approach_region_score",
            "landing_region_score",
            "progress",
        ]:
            if key in metrics:
                return float(metrics.get(key, 0.0))
        return 0.0

    @staticmethod
    def _event_rates(steps: List[Dict[str, Any]]) -> Dict[str, float]:
        if not steps:
            return {}
        totals: Dict[str, int] = {}
        for step in steps:
            for key, value in step.get("events", {}).items():
                totals[key] = totals.get(key, 0) + int(bool(value))
        return {key: value / max(1, len(steps)) for key, value in totals.items()}

    @staticmethod
    def _max_event_toggle_count(steps: List[Dict[str, Any]]) -> int:
        counts: Dict[str, int] = {}
        last_values: Dict[str, bool] = {}
        for step in steps:
            for key, value in step.get("events", {}).items():
                value = bool(value)
                if key in last_values and last_values[key] != value:
                    counts[key] = counts.get(key, 0) + 1
                last_values[key] = value
        return max(counts.values()) if counts else 0

    @staticmethod
    def _first_info(info: Any) -> Dict[str, Any]:
        if isinstance(info, list) and info:
            return info[0] if isinstance(info[0], dict) else {}
        if isinstance(info, tuple) and info:
            return info[0] if isinstance(info[0], dict) else {}
        return info if isinstance(info, dict) else {}

    @staticmethod
    def _to_float(value: Any) -> float:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else 0.0

    @staticmethod
    def _to_list(value: Any) -> Any:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()

    @staticmethod
    def _float_dict(data: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in data.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _bool_dict(data: Dict[str, Any]) -> Dict[str, bool]:
        return {str(key): bool(value) for key, value in data.items()}
