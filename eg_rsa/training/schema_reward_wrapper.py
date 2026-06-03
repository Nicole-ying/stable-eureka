from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from eg_rsa.diagnostics.event_evaluator import EventEvaluator
from eg_rsa.diagnostics.task_metrics import TaskMetricEvaluator
from eg_rsa.env_adapters.box_obs_adapter import BoxObsAdapter
from eg_rsa.reward.schema import RewardSchema


class SchemaRewardWrapper(gym.Wrapper):
    """Replace environment reward with EG-RSA schema reward.

    This wrapper does not use the original reward for training.  The original
    reward is preserved in `info['oracle_reward_posthoc']` only for later
    reporting.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_schema: RewardSchema,
        obs_adapter: BoxObsAdapter,
        task_metric_evaluator: TaskMetricEvaluator,
        event_evaluator: EventEvaluator,
    ):
        super().__init__(env)
        self.reward_schema = reward_schema
        self.obs_adapter = obs_adapter
        self.task_metric_evaluator = task_metric_evaluator
        self.event_evaluator = event_evaluator

    def step(self, action: Any):
        obs, oracle_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        obs_map = self.obs_adapter.obs_to_map(obs)
        events = self.event_evaluator.evaluate(obs_map, action)
        task_metrics = self.task_metric_evaluator.evaluate(obs_map, action, events)
        reward, components = self._compute_schema_reward(obs_map, action, events)
        info["oracle_reward_posthoc"] = float(oracle_reward)
        info["components"] = components
        info["task_metrics"] = task_metrics
        info["events"] = events
        info["obs_map"] = obs_map
        return obs, float(reward), terminated, truncated, info

    def _compute_schema_reward(
        self,
        obs_map: Dict[str, float],
        action: Optional[Any],
        events: Dict[str, bool],
    ) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        components: Dict[str, float] = {}
        for component in self.reward_schema.components:
            if not component.enabled:
                continue
            raw = self._component_raw(component.type, component.inputs, component.params, obs_map, action, events)
            if component.clip is not None:
                raw = min(max(raw, float(component.clip[0])), float(component.clip[1]))
            value = float(component.weight) * float(raw)
            components[component.name] = value
            total += value
        for rule in self.reward_schema.event_rules:
            if not rule.enabled:
                continue
            ok = True
            for key, expected in rule.condition.items():
                if key == "duration_steps":
                    continue
                ok = ok and (events.get(key, False) == expected)
            value = float(rule.weight) if ok else 0.0
            components[rule.name] = value
            total += value
        components["reward"] = float(total)
        return float(total), components

    @staticmethod
    def _component_raw(typ, inputs, params, obs_map, action, events) -> float:
        import math
        import numpy as np

        if typ == "distance_penalty":
            target = params.get("target", [0.0 for _ in inputs])
            return -math.sqrt(sum((float(obs_map.get(k, 0.0)) - float(target[i])) ** 2 for i, k in enumerate(inputs)))
        if typ == "distance_progress":
            return float(obs_map.get(params.get("progress_key", "progress"), 0.0))
        if typ == "angle_penalty":
            return -sum(abs(float(obs_map.get(k, 0.0))) for k in inputs)
        if typ == "velocity_penalty":
            return -math.sqrt(sum(float(obs_map.get(k, 0.0)) ** 2 for k in inputs))
        if typ == "action_penalty":
            return 0.0 if action is None else -float(np.sum(np.square(np.asarray(action, dtype=float))))
        if typ == "constant_alive":
            return float(params.get("value", 1.0))
        if typ == "event_bonus":
            return 1.0 if bool(events.get(params.get("event", ""), False)) else 0.0
        return 0.0
