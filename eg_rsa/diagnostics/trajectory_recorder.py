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
            if seed is not None and hasattr(env, "reset"):
                reset_out = env.reset(seed=seed + episode_id)
            else:
                reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            steps = []
            total_reward = 0.0
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
                obs_map = self.obs_adapter.obs_to_map(obs)
                events = self.event_evaluator.evaluate(obs_map, action)
                task_metrics = self.task_metric_evaluator.evaluate(obs_map, action, events)
                components = self._extract_components(info_dict)
                steps.append(
                    {
                        "t": t,
                        "obs": self._to_list(obs),
                        "action": self._to_list(action),
                        "reward": reward_float,
                        "components": components,
                        "task_metrics": task_metrics,
                        "events": events,
                        "done": done,
                    }
                )
                total_reward += reward_float
                obs = next_obs
                t += 1
            trajectories.append(
                {
                    "episode_id": episode_id,
                    "steps": steps,
                    "summary": self._summary(total_reward, steps),
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
        out = {}
        for key, value in components.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _summary(total_reward: float, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not steps:
            return {"episode_reward": total_reward, "episode_length": 0, "progress_score": 0.0, "success": 0.0}
        last_metrics = steps[-1].get("task_metrics", {})
        success = max(float(step.get("task_metrics", {}).get("success", 0.0)) for step in steps)
        return {
            "episode_reward": float(total_reward),
            "episode_length": len(steps),
            "progress_score": float(last_metrics.get("progress", 0.0)),
            "success": float(success),
        }

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
