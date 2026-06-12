from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from eg_rsa.diagnostics.event_evaluator import EventEvaluator
from eg_rsa.diagnostics.task_metrics import TaskMetricEvaluator
from eg_rsa.env_adapters.box_obs_adapter import BoxObsAdapter
from eg_rsa.env_adapters.action_primitive_mapper import ActionPrimitiveMapper
from eg_rsa.reward.formula_ast import eval_formula_ast
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
            expr_ast = condition.get("expr_ast") or condition.get("condition_ast")
            if expr_ast is None:
                base_ok = False
            else:
                base_ok = bool(
                    self._safe_ast(
                        ast_node=expr_ast,
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
            ast_node = params.get("formula_ast")
            if ast_node is None:
                raise ValueError(f"formula_component {name} requires params.formula_ast")
            return float(self._safe_ast(
                ast_node=ast_node,
                obs_map=obs_map,
                action=action,
                events=events,
                task_metrics=task_metrics,
            ))

        if typ == "conditional_formula_component":
            condition_ast = params.get("condition_ast")
            formula_ast = params.get("formula_ast")
            if condition_ast is None or formula_ast is None:
                raise ValueError(f"conditional_formula_component {name} requires condition_ast and formula_ast")
            cond = self._safe_ast(
                ast_node=condition_ast,
                obs_map=obs_map,
                action=action,
                events=events,
                task_metrics=task_metrics,
            )
            if bool(cond):
                return float(self._safe_ast(
                    ast_node=formula_ast,
                    obs_map=obs_map,
                    action=action,
                    events=events,
                    task_metrics=task_metrics,
                ))
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
            ast_node = params.get("formula_ast")
            if ast_node is not None:
                return float(self._safe_ast(
                    ast_node=ast_node,
                    obs_map=obs_map,
                    action=action,
                    events=events,
                    task_metrics=task_metrics,
                ))
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

    def _safe_ast(
        self,
        ast_node: Any,
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
        return eval_formula_ast(ast_node, variables=variables)

    def _primitive_vars(
        self,
        obs_map: Dict[str, float],
        action: Optional[Any],
        events: Dict[str, bool],
        task_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build the variable namespace used by AST reward formulas.

        This must be environment-agnostic. Earlier versions hard-coded names such
        as x/vx/angle, which broke source-aware bootstrap when the LLM inferred
        valid but different names such as horiz_pos. The canonical source of
        truth is now the generated primitive interface -> observation_mapping ->
        BoxObsAdapter -> obs_map. Therefore every obs_map key must be exposed to
        FormulaAST.eval exactly as-is.
        """
        variables: Dict[str, Any] = {}

        # 1. Observation primitive variables from generated observation_mapping.
        for key, value in (obs_map or {}).items():
            variables[str(key)] = float(value)

        # 2. Diagnostic task metrics are available for metric-based schemas but
        # must not overwrite primitive observation variables with the same name.
        for key, value in (task_metrics or {}).items():
            variables.setdefault(str(key), float(value))

        # 3. Event booleans are also exposed by name for event_predicate ASTs.
        for key, value in (events or {}).items():
            variables.setdefault(str(key), bool(value))

        # 4. Action primitive variables from generated action_mapping.
        action_vars = self.action_mapper.map(action)
        for key, value in action_vars.items():
            variables[str(key)] = float(value)

        # 5. Backward-compatible aliases for older LunarLander-style schemas.
        # These aliases are added only when absent; they should not constrain new
        # source-aware variable names.
        def get_float(name: str, default: float = 0.0) -> float:
            if name in variables:
                return float(variables.get(name, default))
            return float(default)

        def get_bool(name: str, default: bool = False) -> bool:
            if name in variables:
                try:
                    return bool(float(variables[name]) > 0.5)
                except Exception:
                    return bool(variables[name])
            return bool(default)

        legacy_defaults = {
            "x": get_float("x"),
            "y": get_float("y"),
            "vx": get_float("vx"),
            "vy": get_float("vy"),
            "angle": get_float("angle"),
            "angular_velocity": get_float("angular_velocity"),
            "left_contact": get_bool("left_contact") or get_bool("left_leg_contact"),
            "right_contact": get_bool("right_contact") or get_bool("right_leg_contact"),
            "contact": get_bool("contact"),
            "both_contact": get_bool("both_contact"),
        }
        for key, value in legacy_defaults.items():
            variables.setdefault(key, value)

        return variables

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
