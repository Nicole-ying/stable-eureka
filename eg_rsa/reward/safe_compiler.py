from __future__ import annotations

import json

from eg_rsa.reward.schema import RewardSchema
from eg_rsa.reward.formula_validator import FormulaValidator


class SafeRewardCompiler:
    """Safer compact compiler for EG-RSA reward schemas.

    The exported reward code mirrors SchemaRewardWrapper semantics for:
      - metric_delta;
      - metric_stagnation_penalty;
      - one_time event rules;
      - event-rule duration_steps.
    """

    SUPPORTED_COMPONENTS = {
        "distance_penalty",
        "distance_progress",
        "angle_penalty",
        "velocity_penalty",
        "action_penalty",
        "constant_alive",
        "event_bonus",
        "metric_value",
        "metric_delta",
        "metric_threshold_bonus",
        "metric_stagnation_penalty",
        "formula_component",
        "conditional_formula_component",
    }

    @staticmethod
    def compile(schema: RewardSchema, function_name: str = "compute_reward", indent: str = "    ") -> str:
        SafeRewardCompiler._validate(schema)
        schema_text = json.dumps(schema.to_dict(), ensure_ascii=False)
        schema_literal = repr(schema_text)
        i = indent
        return f'''# Generated code by EG-RSA
{i}def {function_name}(self, obs_map, action=None, state_flags=None):
{i}    import json
{i}    import math
{i}    import numpy as np
{i}    schema = json.loads({schema_literal})
{i}    state_flags = state_flags or {{}}
{i}    if not hasattr(self, "_eg_rsa_fired_event_rules"):
{i}        self._eg_rsa_fired_event_rules = set()
{i}    if not hasattr(self, "_eg_rsa_event_rule_duration_counts"):
{i}        self._eg_rsa_event_rule_duration_counts = {{}}
{i}    if not hasattr(self, "_eg_rsa_metric_stagnation_counts"):
{i}        self._eg_rsa_metric_stagnation_counts = {{}}
{i}    task_metrics = {{}}
{i}    prev_task_metrics = {{}}
{i}    if isinstance(obs_map, dict):
{i}        task_metrics = obs_map.get("task_metrics", {{}}) or {{}}
{i}        prev_task_metrics = obs_map.get("_prev_task_metrics", {{}}) or {{}}
{i}    if hasattr(self, "task_metrics"):
{i}        task_metrics = getattr(self, "task_metrics") or task_metrics
{i}    if hasattr(self, "_prev_task_metrics"):
{i}        prev_task_metrics = getattr(self, "_prev_task_metrics") or prev_task_metrics
{i}    individual_reward = {{}}
{i}    total_reward = 0.0
{i}    def _get(name, default=0.0):
{i}        if isinstance(obs_map, dict) and name in obs_map:
{i}            return float(obs_map.get(name, default))
{i}        if name in task_metrics:
{i}            return float(task_metrics.get(name, default))
{i}        return float(default)
{i}    def _action_array():
{i}        if action is None:
{i}            return np.asarray([], dtype=float)
{i}        return np.asarray(action, dtype=float).reshape(-1)
{i}    def _map_action_primitives():
{i}        metadata = schema.get("metadata", {{}}) or {{}}
{i}        mapping = metadata.get("action_mapping", {{}}) or {{}}
{i}        action_variables = metadata.get("action_variables", []) or []
{i}        arr = _action_array()
{i}        out = {{}}
{i}        mapping_type = mapping.get("type")
{i}        if mapping_type == "discrete_lookup":
{i}            a = int(round(float(arr[0]))) if arr.size else 0
{i}            for name, table in (mapping.get("variables", {{}}) or {{}}).items():
{i}                if not isinstance(table, dict):
{i}                    continue
{i}                default = float(table.get("default", 0.0) or 0.0)
{i}                value = table.get(str(a), table.get(a, default))
{i}                out[str(name)] = float(value)
{i}            return out
{i}        if mapping_type == "continuous_indices":
{i}            for name, index in (mapping.get("variables", {{}}) or {{}}).items():
{i}                try:
{i}                    idx = int(index)
{i}                except Exception:
{i}                    out[str(name)] = 0.0
{i}                    continue
{i}                out[str(name)] = float(arr[idx]) if idx < arr.size else 0.0
{i}            return out
{i}        names = [
{i}            str(item.get("name"))
{i}            for item in action_variables
{i}            if isinstance(item, dict) and item.get("name")
{i}        ]
{i}        if set(names) >= {{"main_engine", "side_engine"}} and arr.size == 1:
{i}            a = int(round(float(arr[0])))
{i}            return {{
{i}                "main_engine": 1.0 if a == 2 else 0.0,
{i}                "side_engine": -1.0 if a == 1 else (1.0 if a == 3 else 0.0),
{i}            }}
{i}        for idx, value in enumerate(arr):
{i}            out[f"action_{{idx}}"] = float(value)
{i}        for idx, name in enumerate(names):
{i}            out[name] = float(arr[idx]) if idx < arr.size else 0.0
{i}        return out
{i}    def _clip_fn(value, low, high):
{i}        return float(np.clip(float(value), float(low), float(high)))
{i}    _allowed_formula_functions = {{
{i}        "abs": abs,
{i}        "min": min,
{i}        "max": max,
{i}        "sqrt": math.sqrt,
{i}        "exp": math.exp,
{i}        "tanh": math.tanh,
{i}        "clip": _clip_fn,
{i}    }}
{i}    def _primitive_vars():
{i}        variables = {{
{i}            "x": _get("x", 0.0),
{i}            "y": _get("y", 0.0),
{i}            "vx": _get("vx", 0.0),
{i}            "vy": _get("vy", 0.0),
{i}            "angle": _get("angle", 0.0),
{i}            "angular_velocity": _get("angular_velocity", _get("angularVelocity", 0.0)),
{i}            "left_contact": bool(obs_map.get("left_contact", obs_map.get("leftContact", False))) if isinstance(obs_map, dict) else False,
{i}            "right_contact": bool(obs_map.get("right_contact", obs_map.get("rightContact", False))) if isinstance(obs_map, dict) else False,
{i}            "contact": bool(obs_map.get("contact", False)) if isinstance(obs_map, dict) else False,
{i}            "both_contact": bool(obs_map.get("both_contact", False)) if isinstance(obs_map, dict) else False,
{i}        }}
{i}        variables.update(_map_action_primitives())
{i}        return variables
{i}    def _safe_formula(expr):
{i}        if not isinstance(expr, str) or not expr.strip():
{i}            return 0.0
{i}        safe_locals = {{}}
{i}        safe_locals.update(_allowed_formula_functions)
{i}        safe_locals.update(_primitive_vars())
{i}        return float(eval(compile(expr, "<eg_rsa_formula>", "eval"), {{"__builtins__": {{}}}}, safe_locals))
{i}    def _clip(value, clip_range):
{i}        if clip_range is None:
{i}            return float(value)
{i}        return float(np.clip(value, clip_range[0], clip_range[1]))
{i}    for component in schema.get("components", []):
{i}        if not component.get("enabled", True):
{i}            continue
{i}        name = component["name"]
{i}        ctype = component["type"]
{i}        weight = float(component.get("weight", 1.0))
{i}        inputs = component.get("inputs", [])
{i}        params = component.get("params", {{}})
{i}        raw = 0.0
{i}        if ctype == "distance_penalty":
{i}            target = params.get("target", [0.0 for _ in inputs])
{i}            raw = -math.sqrt(sum((_get(key) - float(target[idx])) ** 2 for idx, key in enumerate(inputs)))
{i}        elif ctype == "distance_progress":
{i}            raw = _get(params.get("progress_key", "progress"))
{i}        elif ctype == "angle_penalty":
{i}            raw = -sum(abs(_get(key)) for key in inputs)
{i}        elif ctype == "velocity_penalty":
{i}            raw = -math.sqrt(sum(_get(key) ** 2 for key in inputs))
{i}        elif ctype == "action_penalty":
{i}            formula = params.get("formula")
{i}            if isinstance(formula, str) and formula.strip():
{i}                raw = _safe_formula(formula)
{i}            else:
{i}                raw = 0.0 if action is None else -float(np.sum(np.square(np.asarray(action, dtype=float))))
{i}        elif ctype == "constant_alive":
{i}            raw = float(params.get("value", 1.0))
{i}        elif ctype == "event_bonus":
{i}            raw = 1.0 if bool(state_flags.get(params.get("event", name), False)) else 0.0
{i}        elif ctype == "metric_value":
{i}            raw = float(task_metrics.get(params.get("metric", ""), 0.0))
{i}        elif ctype == "metric_delta":
{i}            metric = params.get("metric", "")
{i}            current = float(task_metrics.get(metric, 0.0))
{i}            previous = float(prev_task_metrics.get(metric, current))
{i}            delta = current - previous
{i}            if bool(params.get("positive_only", True)):
{i}                delta = max(0.0, delta)
{i}            raw = float(delta)
{i}        elif ctype == "metric_threshold_bonus":
{i}            metric = params.get("metric", "")
{i}            threshold = float(params.get("threshold", 0.0))
{i}            direction = params.get("direction", "ge")
{i}            value = float(task_metrics.get(metric, 0.0))
{i}            if direction == "le":
{i}                raw = 1.0 if value <= threshold else 0.0
{i}            else:
{i}                raw = 1.0 if value >= threshold else 0.0
{i}        elif ctype == "metric_stagnation_penalty":
{i}            metric = params.get("metric", "")
{i}            threshold = float(params.get("threshold", 1e-3))
{i}            window = int(params.get("window", 20))
{i}            current = float(task_metrics.get(metric, 0.0))
{i}            previous = float(prev_task_metrics.get(metric, current))
{i}            delta = abs(current - previous)
{i}            if delta < threshold:
{i}                self._eg_rsa_metric_stagnation_counts[name] = self._eg_rsa_metric_stagnation_counts.get(name, 0) + 1
{i}            else:
{i}                self._eg_rsa_metric_stagnation_counts[name] = 0
{i}            raw = -1.0 if self._eg_rsa_metric_stagnation_counts.get(name, 0) >= window else 0.0
{i}        elif ctype == "formula_component":
{i}            raw = _safe_formula(component.get("formula") or params.get("formula", "0.0"))
{i}        elif ctype == "conditional_formula_component":
{i}            cond_expr = component.get("condition") or params.get("condition", "False")
{i}            raw = _safe_formula(component.get("formula") or params.get("formula", "0.0")) if bool(_safe_formula(cond_expr)) else 0.0
{i}        value = weight * _clip(raw, component.get("clip"))
{i}        individual_reward[name] = float(value)
{i}        total_reward += float(value)
{i}    for rule in schema.get("event_rules", []):
{i}        if not rule.get("enabled", True):
{i}            continue
{i}        name = rule["name"]
{i}        weight = float(rule.get("weight", 1.0))
{i}        rtype = rule.get("type", "event_bonus")
{i}        condition = rule.get("condition", {{}})
{i}        duration_steps = int(condition.get("duration_steps", 1) or 1) if isinstance(condition, dict) else 1
{i}        if rtype == "event_predicate":
{i}            expr = condition.get("expression") or condition.get("formula") or "False"
{i}            base_ok = bool(_safe_formula(expr))
{i}        else:
{i}            base_ok = True
{i}            for key, expected in condition.items():
{i}                if key == "duration_steps":
{i}                    continue
{i}                base_ok = base_ok and (state_flags.get(key, False) == expected)
{i}        if base_ok:
{i}            self._eg_rsa_event_rule_duration_counts[name] = self._eg_rsa_event_rule_duration_counts.get(name, 0) + 1
{i}        else:
{i}            self._eg_rsa_event_rule_duration_counts[name] = 0
{i}        ok = base_ok if duration_steps <= 1 else (base_ok and self._eg_rsa_event_rule_duration_counts.get(name, 0) >= duration_steps)
{i}        if rule.get("one_time", False) and name in self._eg_rsa_fired_event_rules:
{i}            value = 0.0
{i}        else:
{i}            value = float(weight) if ok else 0.0
{i}            if rule.get("one_time", False) and ok:
{i}                self._eg_rsa_fired_event_rules.add(name)
{i}        individual_reward[name] = float(value)
{i}        total_reward += float(value)
{i}    individual_reward["reward"] = float(total_reward)
{i}    return float(total_reward), individual_reward
'''

    @staticmethod
    def _validate(schema: RewardSchema) -> None:
        names = set()
        for component in schema.components:
            if component.name in names:
                raise ValueError(f"Duplicate reward component name: {component.name}")
            names.add(component.name)
            if component.type not in SafeRewardCompiler.SUPPORTED_COMPONENTS:
                raise ValueError(
                    f"Unsupported component type: {component.type}. "
                    f"Supported: {sorted(SafeRewardCompiler.SUPPORTED_COMPONENTS)}"
                )
            if component.type == "action_penalty" and component.params.get("formula"):
                result = FormulaValidator.validate_expression(
                    str(component.params.get("formula")),
                    allowed_variables={
                        "x", "y", "vx", "vy", "angle", "angular_velocity",
                        "left_contact", "right_contact", "main_engine", "side_engine",
                        "contact", "both_contact",
                    },
                    allowed_functions={"abs", "min", "max", "sqrt", "exp", "tanh", "clip"},
                )
                if not result.ok:
                    raise ValueError(f"Unsafe action_penalty formula for {component.name}: {result.errors}")

            if component.type in {"formula_component", "conditional_formula_component"}:
                formula = component.params.get("formula")
                if not isinstance(formula, str) or not formula.strip():
                    raise ValueError(f"{component.type} {component.name} requires params.formula")
                result = FormulaValidator.validate_expression(
                    formula,
                    allowed_variables={
                        "x", "y", "vx", "vy", "angle", "angular_velocity",
                        "left_contact", "right_contact", "main_engine", "side_engine",
                        "contact", "both_contact",
                    },
                    allowed_functions={"abs", "min", "max", "sqrt", "exp", "tanh", "clip"},
                )
                if not result.ok:
                    raise ValueError(f"Unsafe formula for {component.name}: {result.errors}")
                if component.type == "conditional_formula_component":
                    condition = component.params.get("condition")
                    if not isinstance(condition, str) or not condition.strip():
                        raise ValueError(f"conditional_formula_component {component.name} requires params.condition")
                    result = FormulaValidator.validate_expression(
                        condition,
                        allowed_variables={
                            "x", "y", "vx", "vy", "angle", "angular_velocity",
                            "left_contact", "right_contact", "main_engine", "side_engine",
                        },
                        allowed_functions={"abs", "min", "max", "sqrt", "exp", "tanh", "clip"},
                    )
                    if not result.ok:
                        raise ValueError(f"Unsafe condition for {component.name}: {result.errors}")
            if component.clip is not None:
                if len(component.clip) != 2 or component.clip[0] > component.clip[1]:
                    raise ValueError(f"Invalid clip range for {component.name}: {component.clip}")
        for rule in schema.event_rules:
            if rule.name in names:
                raise ValueError(f"Duplicate reward item name: {rule.name}")
            names.add(rule.name)
            if rule.type not in {"event_bonus", "event_predicate"}:
                raise ValueError(f"Unsupported event rule type: {rule.type}")
            if not isinstance(rule.condition, dict) or not rule.condition:
                raise ValueError(f"Event rule {rule.name} must have a non-empty condition")
            if rule.type == "event_predicate":
                expr = rule.condition.get("expression") or rule.condition.get("formula")
                if not isinstance(expr, str) or not expr.strip():
                    raise ValueError(f"event_predicate {rule.name} requires condition.expression")
                result = FormulaValidator.validate_expression(
                    expr,
                    allowed_variables={
                        "x", "y", "vx", "vy", "angle", "angular_velocity",
                        "left_contact", "right_contact", "main_engine", "side_engine",
                        "contact", "both_contact",
                    },
                    allowed_functions={"abs", "min", "max", "sqrt", "exp", "tanh", "clip"},
                )
                if not result.ok:
                    raise ValueError(f"Unsafe event predicate for {rule.name}: {result.errors}")
            if "duration_steps" in rule.condition and int(rule.condition["duration_steps"]) <= 0:
                raise ValueError(f"Event rule {rule.name} has invalid duration_steps")
