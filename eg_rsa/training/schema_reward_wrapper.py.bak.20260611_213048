from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from eg_rsa.diagnostics.event_evaluator import EventEvaluator
from eg_rsa.diagnostics.task_metrics import TaskMetricEvaluator
from eg_rsa.env_adapters.box_obs_adapter import BoxObsAdapter
from eg_rsa.env_adapters.action_primitive_mapper import ActionPrimitiveMapper
from eg_rsa.reward.safe_formula_eval import safe_eval_formula
from eg_rsa.reward.schema import RewardSchema


class SchemaRewardWrapper(gym.Wrapper):
    """Replace environment reward with EG-RSA schema reward.

    This wrapper does not use the original reward for training. The original
    reward is preserved in `info['oracle_reward_posthoc']` only for later
    reporting.

    V2 note:
        Training uses this wrapper directly. `compiled_reward.py` is only an
        artifact for inspection. Therefore formula_component and event_predicate
        must be supported here, not only in SafeRewardCompiler.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_schema: RewardSchema,
        obs_adapter: BoxObsAdapter,
        task_metric_evaluator: TaskMetricEvaluator,
        event_evaluator: EventEvaluator,
        action_mapper: ActionPrimitiveMapper | None = None,
    ):
        super().__init__(env)
        self.reward_schema = reward_schema
        self.obs_adapter = obs_adapter
        self.task_metric_evaluator = task_metric_evaluator
        self.event_evaluator = event_evaluator
        self.action_mapper = action_mapper or ActionPrimitiveMapper(
            mapping_spec=(reward_schema.metadata or {}).get("action_mapping", {}),
            action_variables=(reward_schema.metadata or {}).get("action_variables", []),
        )
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

        reward, components = self._compute_schema_reward(
            obs_map=obs_map,
            action=action,
            events=events,
            task_metrics=task_metrics,
        )

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

            raw = self._component_raw(
                name=component.name,
                typ=component.type,
                inputs=component.inputs,
                params=component.params,
                obs_map=obs_map,
                action=action,
                events=events,
                task_metrics=task_metrics,
            )

            if component.clip is not None:
                raw = min(max(raw, float(component.clip[0])), float(component.clip[1]))

            value = float(component.weight) * float(raw)
            components[component.name] = value
            total += value

        for rule in self.reward_schema.event_rules:
            if not rule.enabled:
                continue

            ok = self._event_rule_condition_ok(
                rule_name=rule.name,
                rule_type=rule.type,
                condition=rule.condition,
                obs_map=obs_map,
                action=action,
                events=events,
                task_metrics=task_metrics,
            )

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

    def _event_rule_condition_ok(
        self,
        rule_name: str,
        rule_type: str,
        condition: Dict[str, Any],
        obs_map: Dict[str, float],
        action: Optional[Any],
        events: Dict[str, bool],
        task_metrics: Dict[str, float],
    ) -> bool:
        condition = dict(condition or {})
        duration_steps = int(condition.get("duration_steps", 1) or 1)

        if rule_type == "event_predicate":
            expr = condition.get("expression") or condition.get("formula")
            if not expr:
                base_ok = False
            else:
                base_ok = bool(
                    self._safe_formula(
                        expr=str(expr),
                        obs_map=obs_map,
                        action=action,
                        events=events,
                        task_metrics=task_metrics,
                    )
                )
        else:
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
            return bool(base_ok)

        return bool(base_ok and self._event_rule_duration_counts.get(rule_name, 0) >= duration_steps)

    def _component_raw(
        self,
        name: str,
        typ: str,
        inputs: Any,
        params: Dict[str, Any],
        obs_map: Dict[str, float],
        action: Optional[Any],
        events: Dict[str, bool],
        task_metrics: Dict[str, float],
    ) -> float:
        import math

        params = dict(params or {})

        if typ == "formula_component":
            return self._safe_formula(
                expr=str(params.get("formula", "0.0")),
                obs_map=obs_map,
                action=action,
                events=events,
                task_metrics=task_metrics,
            )

        if typ == "conditional_formula_component":
            cond = self._safe_formula(
                expr=str(params.get("condition", "False")),
                obs_map=obs_map,
                action=action,
                events=events,
                task_metrics=task_metrics,
            )
            if bool(cond):
                return self._safe_formula(
                    expr=str(params.get("formula", "0.0")),
                    obs_map=obs_map,
                    action=action,
                    events=events,
                    task_metrics=task_metrics,
                )
            return 0.0

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
            formula = params.get("formula")
            if isinstance(formula, str) and formula.strip():
                return self._safe_formula(
                    expr=formula,
                    obs_map=obs_map,
                    action=action,
                    events=events,
                    task_metrics=task_metrics,
                )
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

    def _safe_formula(
        self,
        expr: str,
        obs_map: Dict[str, float],
        action: Optional[Any],
        events: Dict[str, bool],
        task_metrics: Dict[str, float],
    ) -> float:
        variables = self._primitive_vars(
            obs_map=obs_map,
            action=action,
            events=events,
            task_metrics=task_metrics,
        )
        return safe_eval_formula(expr, variables=variables)

    def _primitive_vars(
        self,
        obs_map: Dict[str, float],
        action: Optional[Any],
        events: Dict[str, bool],
        task_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        def get_float(name: str, default: float = 0.0) -> float:
            if name in obs_map:
                return float(obs_map.get(name, default))
            if name in task_metrics:
                return float(task_metrics.get(name, default))
            return float(default)

        def get_bool(event_name: str, obs_name: str, default: bool = False) -> bool:
            if event_name in events:
                return bool(events[event_name])
            if event_name in obs_map:
                return bool(float(obs_map[event_name]) > 0.5)
            if obs_name in obs_map:
                return bool(float(obs_map[obs_name]) > 0.5)
            return bool(default)

        action_vars = self.action_mapper.map(action)

        variables = {
            "x": get_float("x"),
            "y": get_float("y"),
            "vx": get_float("vx"),
            "vy": get_float("vy"),
            "angle": get_float("angle"),
            "angular_velocity": get_float("angular_velocity"),
            "left_contact": get_bool("left_contact", "left_leg_contact"),
            "right_contact": get_bool("right_contact", "right_leg_contact"),
            "contact": bool(events.get("contact", False)),
            "both_contact": bool(events.get("both_contact", False)),
        }
        variables.update({str(k): float(v) for k, v in action_vars.items()})
        return variables

    @staticmethod
    def _action_to_engine_vars(action: Optional[Any]) -> Tuple[float, float]:
        """LEGACY fallback only. Active runtime uses ActionPrimitiveMapper.

        Map LunarLander actions to primitive engine variables.

        Discrete LunarLander:
            0 = do nothing
            1 = fire left orientation engine
            2 = fire main engine
            3 = fire right orientation engine

        Continuous LunarLander:
            action[0] = main engine signal
            action[1] = side engine signal
        """

        if action is None:
            return 0.0, 0.0

        arr = np.asarray(action, dtype=float).reshape(-1)
        if arr.size == 0:
            return 0.0, 0.0

        if arr.size == 1:
            a = int(round(float(arr[0])))
            main_engine = 1.0 if a == 2 else 0.0
            if a == 1:
                side_engine = -1.0
            elif a == 3:
                side_engine = 1.0
            else:
                side_engine = 0.0
            return main_engine, side_engine

        return float(arr[0]), float(arr[1])
