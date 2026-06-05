from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from eg_rsa.diagnostics.event_evaluator import EventEvaluator
from eg_rsa.diagnostics.task_metrics import TaskMetricEvaluator
from eg_rsa.env_adapters.box_obs_adapter import BoxObsAdapter
from eg_rsa.reward.schema import RewardSchema


class SchemaRewardWrapper(gym.Wrapper):
    """Replace environment reward with EG-RSA schema reward.

    This wrapper does not use the original reward for training. The original
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
        self._fired_event_rules = set()
        self._event_rule_duration_counts: Dict[str, int] = {}
        self._prev_task_metrics: Dict[str, float] = {}
        self._metric_stagnation_counts: Dict[str, int] = {}

    def reset(self, **kwargs):
        self._fired_event_rules = set()
        self._event_rule_duration_counts = {}
        self._prev_task_metrics = {}
        self._metric_stagnation_counts = {}
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, oracle_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        obs_map = self.obs_adapter.obs_to_map(obs)
        events = self.event_evaluator.evaluate(obs_map, action)
        task_metrics = self.task_metric_evaluator.evaluate(obs_map, action, events)
        reward, components = self._compute_schema_reward(obs_map, action, events, task_metrics)
        self._prev_task_metrics = dict(task_metrics)
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
        task_metrics: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        components: Dict[str, float] = {}
        for component in self.reward_schema.components:
            if not component.enabled:
                continue
            raw = self._component_raw(component.name, component.type, component.inputs, component.params, obs_map, action, events, task_metrics)
            if component.clip is not None:
                raw = min(max(raw, float(component.clip[0])), float(component.clip[1]))
            value = float(component.weight) * float(raw)
            components[component.name] = value
            total += value
        for rule in self.reward_schema.event_rules:
            if not rule.enabled:
                continue
            ok = self._event_rule_condition_ok(rule.name, rule.condition, events)
            if rule.one_time and rule.name in self._fired_event_rules:
                value = 0.0
            else:
                value = float(rule.weight) if ok else 0.0
                if rule.one_time and ok:
                    self._fired_event_rules.add(rule.name)
            components[rule.name] = value
            total += value
        components["reward"] = float(total)
        return float(total), components

    def _event_rule_condition_ok(self, rule_name: str, condition: Dict[str, Any], events: Dict[str, bool]) -> bool:
        duration_steps = int(condition.get("duration_steps", 1) or 1)
        base_ok = True
        for key, expected in condition.items():
            if key == "duration_steps":
                continue
            base_ok = base_ok and (events.get(key, False) == expected)
        if base_ok:
            self._event_rule_duration_counts[rule_name] = self._event_rule_duration_counts.get(rule_name, 0) + 1
        else:
            self._event_rule_duration_counts[rule_name] = 0
        if duration_steps <= 1:
            return base_ok
        return base_ok and self._event_rule_duration_counts.get(rule_name, 0) >= duration_steps

    def _component_raw(self, name, typ, inputs, params, obs_map, action, events, task_metrics) -> float:
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
        if typ == "metric_value":
            return float(task_metrics.get(params.get("metric", ""), 0.0))
        if typ == "metric_delta":
            metric = params.get("metric", "")
            current = float(task_metrics.get(metric, 0.0))
            previous = float(self._prev_task_metrics.get(metric, current))
            delta = current - previous
            if bool(params.get("positive_only", True)):
                delta = max(0.0, delta)
            return float(delta)
        if typ == "metric_threshold_bonus":
            metric = params.get("metric", "")
            threshold = float(params.get("threshold", 0.0))
            direction = params.get("direction", "ge")
            value = float(task_metrics.get(metric, 0.0))
            if direction == "le":
                return 1.0 if value <= threshold else 0.0
            return 1.0 if value >= threshold else 0.0
        if typ == "metric_stagnation_penalty":
            metric = params.get("metric", "")
            threshold = float(params.get("threshold", 1e-3))
            window = int(params.get("window", 20))
            current = float(task_metrics.get(metric, 0.0))
            previous = float(self._prev_task_metrics.get(metric, current))
            delta = abs(current - previous)
            key = name
            if delta < threshold:
                self._metric_stagnation_counts[key] = self._metric_stagnation_counts.get(key, 0) + 1
            else:
                self._metric_stagnation_counts[key] = 0
            return -1.0 if self._metric_stagnation_counts.get(key, 0) >= window else 0.0
        return 0.0
