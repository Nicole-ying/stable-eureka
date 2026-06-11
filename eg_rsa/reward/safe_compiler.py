from __future__ import annotations

import json

from eg_rsa.reward.formula_ast import validate_formula_ast
from eg_rsa.reward.schema import RewardSchema


class SafeRewardCompiler:
    """Generate readable reward artifact from AST-first RewardSchema.

    Training uses SchemaRewardWrapper directly; this compiled_reward.py is an
    inspection artifact, but it mirrors AST runtime semantics.
    """

    SUPPORTED_COMPONENTS = {
        "formula_component",
        "conditional_formula_component",
        "action_penalty",
        "metric_value",
        "metric_delta",
        "metric_threshold_bonus",
        "metric_stagnation_penalty",
        "constant_alive",
        "event_bonus",
    }

    @staticmethod
    def compile(schema: RewardSchema, function_name: str = "compute_reward", indent: str = "    ") -> str:
        SafeRewardCompiler._validate(schema)
        schema_text = json.dumps(schema.to_dict(), ensure_ascii=False)
        schema_literal = repr(schema_text)
        i = indent
        return f'''# Generated code by EG-RSA AST-IR compiler
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
{i}    task_metrics = obs_map.get("task_metrics", {{}}) if isinstance(obs_map, dict) else {{}}
{i}    prev_task_metrics = obs_map.get("_prev_task_metrics", {{}}) if isinstance(obs_map, dict) else {{}}
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
{i}                default = float(table.get("default", 0.0) or 0.0)
{i}                value = table.get(str(a), table.get(a, default))
{i}                out[str(name)] = float(value)
{i}            return out
{i}        for idx, value in enumerate(arr):
{i}            out[f"action_{{idx}}"] = float(value)
{i}        for idx, item in enumerate(action_variables):
{i}            if isinstance(item, dict) and item.get("name"):
{i}                out[str(item["name"])] = float(arr[idx]) if idx < arr.size else 0.0
{i}        return out
{i}    def _vars():
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
{i}    def _eval_ast(node, variables):
{i}        if isinstance(node, (int, float)):
{i}            return float(node)
{i}        if isinstance(node, bool):
{i}            return bool(node)
{i}        if not isinstance(node, dict):
{i}            raise ValueError(f"Bad AST node: {{node!r}}")
{i}        if "var" in node:
{i}            return variables[str(node["var"])]
{i}        if "const" in node:
{i}            return float(node["const"])
{i}        if "bool" in node:
{i}            return bool(node["bool"])
{i}        op = str(node.get("op", "")).lower()
{i}        if op == "add":
{i}            return sum(float(_eval_ast(x, variables)) for x in node["args"])
{i}        if op == "mul":
{i}            v = 1.0
{i}            for x in node["args"]:
{i}                v *= float(_eval_ast(x, variables))
{i}            return v
{i}        if op == "sub":
{i}            return float(_eval_ast(node["left"], variables)) - float(_eval_ast(node["right"], variables))
{i}        if op == "div":
{i}            d = float(_eval_ast(node["right"], variables))
{i}            return 0.0 if abs(d) < 1e-12 else float(_eval_ast(node["left"], variables)) / d
{i}        if op == "neg":
{i}            return -float(_eval_ast(node["arg"], variables))
{i}        if op == "abs":
{i}            return abs(float(_eval_ast(node["arg"], variables)))
{i}        if op == "sqrt":
{i}            return math.sqrt(max(0.0, float(_eval_ast(node["arg"], variables))))
{i}        if op == "exp":
{i}            return math.exp(max(-50.0, min(50.0, float(_eval_ast(node["arg"], variables)))))
{i}        if op == "tanh":
{i}            return math.tanh(float(_eval_ast(node["arg"], variables)))
{i}        if op == "min":
{i}            return min(float(_eval_ast(x, variables)) for x in node["args"])
{i}        if op == "max":
{i}            return max(float(_eval_ast(x, variables)) for x in node["args"])
{i}        if op == "clip":
{i}            value = float(_eval_ast(node["args"][0], variables))
{i}            low = float(_eval_ast(node["args"][1], variables))
{i}            high = float(_eval_ast(node["args"][2], variables))
{i}            if low > high:
{i}                low, high = high, low
{i}            return max(low, min(high, value))
{i}        if op == "and":
{i}            return all(bool(_eval_ast(x, variables)) for x in node["args"])
{i}        if op == "or":
{i}            return any(bool(_eval_ast(x, variables)) for x in node["args"])
{i}        if op == "not":
{i}            return not bool(_eval_ast(node["arg"], variables))
{i}        left = _eval_ast(node.get("left"), variables) if "left" in node else None
{i}        right = _eval_ast(node.get("right"), variables) if "right" in node else None
{i}        if op == "lt":
{i}            return float(left) < float(right)
{i}        if op == "le":
{i}            return float(left) <= float(right)
{i}        if op == "gt":
{i}            return float(left) > float(right)
{i}        if op == "ge":
{i}            return float(left) >= float(right)
{i}        if op == "eq":
{i}            return left == right
{i}        if op == "ne":
{i}            return left != right
{i}        raise ValueError(f"Unsupported AST op: {{op}}")
{i}    def _clip(value, clip_range):
{i}        if clip_range is None:
{i}            return float(value)
{i}        return float(np.clip(value, clip_range[0], clip_range[1]))
{i}    variables = _vars()
{i}    individual_reward = {{}}
{i}    total_reward = 0.0
{i}    for component in schema.get("components", []):
{i}        if not component.get("enabled", True):
{i}            continue
{i}        name = component["name"]
{i}        ctype = component["type"]
{i}        params = component.get("params", {{}}) or {{}}
{i}        weight = float(component.get("weight", 1.0))
{i}        raw = 0.0
{i}        if ctype == "formula_component":
{i}            raw = float(_eval_ast(component.get("formula_ast") or params.get("formula_ast"), variables))
{i}        elif ctype == "conditional_formula_component":
{i}            cond = bool(_eval_ast(component.get("condition_ast") or params.get("condition_ast"), variables))
{i}            raw = float(_eval_ast(component.get("formula_ast") or params.get("formula_ast"), variables)) if cond else 0.0
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
{i}            raw = max(0.0, delta) if bool(params.get("positive_only", True)) else delta
{i}        value = weight * _clip(raw, component.get("clip"))
{i}        individual_reward[name] = float(value)
{i}        total_reward += float(value)
{i}    for rule in schema.get("event_rules", []):
{i}        if not rule.get("enabled", True):
{i}            continue
{i}        name = rule["name"]
{i}        condition = rule.get("condition", {{}}) or {{}}
{i}        duration_steps = int(condition.get("duration_steps", 1) or 1)
{i}        if rule.get("type") == "event_predicate":
{i}            base_ok = bool(_eval_ast(condition.get("expr_ast") or rule.get("condition_ast"), variables))
{i}        else:
{i}            base_ok = True
{i}            for key, expected in condition.items():
{i}                if key == "duration_steps":
{i}                    continue
{i}                base_ok = base_ok and (bool(state_flags.get(key, False)) == bool(expected))
{i}        if base_ok:
{i}            self._eg_rsa_event_rule_duration_counts[name] = self._eg_rsa_event_rule_duration_counts.get(name, 0) + 1
{i}        else:
{i}            self._eg_rsa_event_rule_duration_counts[name] = 0
{i}        ok = base_ok if duration_steps <= 1 else (base_ok and self._eg_rsa_event_rule_duration_counts.get(name, 0) >= duration_steps)
{i}        if rule.get("one_time", False) and name in self._eg_rsa_fired_event_rules:
{i}            value = 0.0
{i}        else:
{i}            value = float(rule.get("weight", 1.0)) if ok else 0.0
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
        allowed_variables = set((schema.metadata or {}).get("allowed_formula_variables", [])) or {
            "x", "y", "vx", "vy", "angle", "angular_velocity",
            "left_contact", "right_contact", "main_engine", "side_engine",
            "contact", "both_contact",
        }

        for component in schema.components:
            if component.name in names:
                raise ValueError(f"Duplicate reward component name: {component.name}")
            names.add(component.name)

            if component.type not in SafeRewardCompiler.SUPPORTED_COMPONENTS:
                raise ValueError(f"Unsupported component type: {component.type}")

            if component.type in {"formula_component", "conditional_formula_component", "action_penalty"}:
                ast_node = component.params.get("formula_ast")
                if ast_node is None:
                    raise ValueError(f"{component.type} {component.name} requires params.formula_ast")
                validation = validate_formula_ast(ast_node, allowed_variables)
                if not validation.ok:
                    raise ValueError(f"Unsafe formula_ast for {component.name}: {validation.errors}")

            if component.type == "conditional_formula_component":
                ast_node = component.params.get("condition_ast")
                if ast_node is None:
                    raise ValueError(f"conditional_formula_component {component.name} requires params.condition_ast")
                validation = validate_formula_ast(ast_node, allowed_variables)
                if not validation.ok:
                    raise ValueError(f"Unsafe condition_ast for {component.name}: {validation.errors}")

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
                ast_node = rule.condition.get("expr_ast")
                if ast_node is None:
                    raise ValueError(f"event_predicate {rule.name} requires condition.expr_ast")
                validation = validate_formula_ast(ast_node, allowed_variables)
                if not validation.ok:
                    raise ValueError(f"Unsafe condition_ast for {rule.name}: {validation.errors}")

            if "duration_steps" in rule.condition and int(rule.condition["duration_steps"]) <= 0:
                raise ValueError(f"Event rule {rule.name} has invalid duration_steps")
