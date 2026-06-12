# Generated code by EG-RSA AST-IR compiler
    def compute_reward(self, obs_map, action=None, state_flags=None):
        import json
        import math
        import numpy as np
        schema = json.loads('{"version": 4, "components": [{"name": "r_center_guidance", "type": "formula_component", "weight": 0.75, "inputs": [], "params": {"formula_ast": {"op": "sub", "left": {"const": 1.0}, "right": {"op": "min", "args": [{"const": 1.0}, {"op": "div", "left": {"op": "abs", "arg": {"var": "x"}}, "right": {"const": 0.2}}]}}}, "clip": [0.0, 1.0], "enabled": true, "semantic_role": "dense_guidance", "reward_timing": "dense", "behavior_channel": "progress", "formula_ast": {"op": "sub", "left": {"const": 1.0}, "right": {"op": "min", "args": [{"const": 1.0}, {"op": "div", "left": {"op": "abs", "arg": {"var": "x"}}, "right": {"const": 0.2}}]}}}, {"name": "r_stability", "type": "formula_component", "weight": 0.5, "inputs": [], "params": {"formula_ast": {"op": "sub", "left": {"const": 0.0}, "right": {"op": "add", "args": [{"op": "abs", "arg": {"var": "angle"}}, {"op": "mul", "args": [{"const": 0.5}, {"op": "abs", "arg": {"var": "angular_velocity"}}]}]}}}, "clip": [-1.0, 0.0], "enabled": true, "semantic_role": "stability_quality", "reward_timing": "dense", "behavior_channel": "safety", "formula_ast": {"op": "sub", "left": {"const": 0.0}, "right": {"op": "add", "args": [{"op": "abs", "arg": {"var": "angle"}}, {"op": "mul", "args": [{"const": 0.5}, {"op": "abs", "arg": {"var": "angular_velocity"}}]}]}}}, {"name": "r_vertical_safety", "type": "conditional_formula_component", "weight": 0.5, "inputs": [], "params": {"formula_ast": {"op": "neg", "arg": {"op": "abs", "arg": {"var": "vy"}}}, "condition_ast": {"op": "lt", "left": {"var": "y"}, "right": {"const": 0.1}}}, "clip": [-1.0, 0.0], "enabled": true, "semantic_role": "safety_constraint", "reward_timing": "dense", "behavior_channel": "safety", "formula_ast": {"op": "neg", "arg": {"op": "abs", "arg": {"var": "vy"}}}, "condition_ast": {"op": "lt", "left": {"var": "y"}, "right": {"const": 0.1}}}, {"name": "r_fuel_penalty", "type": "formula_component", "weight": 1.0, "inputs": [], "params": {"formula_ast": {"op": "mul", "args": [{"const": -0.05}, {"op": "add", "args": [{"var": "main_engine"}, {"op": "abs", "arg": {"var": "side_engine"}}]}]}}, "clip": [-1.0, 0.0], "enabled": true, "semantic_role": "control_cost", "reward_timing": "dense", "behavior_channel": "control", "formula_ast": {"op": "mul", "args": [{"const": -0.05}, {"op": "add", "args": [{"var": "main_engine"}, {"op": "abs", "arg": {"var": "side_engine"}}]}]}}, {"name": "r_descent_guidance", "type": "formula_component", "weight": 0.75, "inputs": [], "params": {"formula_ast": {"op": "sub", "left": {"const": 1.0}, "right": {"var": "y"}}}, "clip": [0.0, 1.0], "enabled": true, "semantic_role": "dense_guidance", "reward_timing": "dense", "behavior_channel": "progress", "formula_ast": {"op": "sub", "left": {"const": 1.0}, "right": {"var": "y"}}}, {"name": "r_speed_penalty", "type": "formula_component", "weight": 0.3, "inputs": [], "params": {"formula_ast": {"op": "mul", "args": [{"const": -0.5}, {"op": "add", "args": [{"op": "abs", "arg": {"var": "vx"}}, {"op": "abs", "arg": {"var": "vy"}}]}]}}, "clip": [-1.0, 0.0], "enabled": true, "semantic_role": "safety_constraint", "reward_timing": "dense", "behavior_channel": "safety", "formula_ast": {"op": "mul", "args": [{"const": -0.5}, {"op": "add", "args": [{"op": "abs", "arg": {"var": "vx"}}, {"op": "abs", "arg": {"var": "vy"}}]}]}}], "event_rules": [{"name": "r_landing_success", "type": "event_predicate", "weight": 200.0, "condition": {"expr_ast": {"op": "and", "args": [{"var": "left_contact"}, {"var": "right_contact"}]}, "duration_steps": 20}, "one_time": true, "enabled": true, "condition_ast": {"op": "and", "args": [{"var": "left_contact"}, {"var": "right_contact"}]}, "semantic_role": "terminal_success", "reward_timing": "sparse_event", "behavior_channel": "completion"}, {"name": "r_crash", "type": "event_predicate", "weight": -100.0, "condition": {"expr_ast": {"op": "and", "args": [{"op": "lt", "left": {"var": "y"}, "right": {"const": 0.0}}, {"op": "not", "arg": {"op": "or", "args": [{"var": "left_contact"}, {"var": "right_contact"}]}}]}, "duration_steps": 1}, "one_time": true, "enabled": true, "condition_ast": {"op": "and", "args": [{"op": "lt", "left": {"var": "y"}, "right": {"const": 0.0}}, {"op": "not", "arg": {"op": "or", "args": [{"var": "left_contact"}, {"var": "right_contact"}]}}]}, "semantic_role": "safety_constraint", "reward_timing": "sparse_event", "behavior_channel": "safety"}], "metadata": {"source": "llm_bootstrap_ast", "formula_ir": "ast", "reward_blueprint_present": true, "action_mapping": {"type": "discrete_lookup", "variables": {"main_engine": {"2": 1.0, "default": 0.0}, "side_engine": {"1": -1.0, "3": 1.0, "default": 0.0}}}, "action_variables": [{"name": "main_engine", "description": "Main engine action or throttle.", "type": "float"}, {"name": "side_engine", "description": "Side engine action or throttle.", "type": "float"}], "allowed_formula_variables": ["x", "y", "vx", "vy", "angle", "angular_velocity", "left_contact", "right_contact", "main_engine", "side_engine"], "allowed_formula_functions": ["abs", "min", "max", "clip", "sqrt", "exp", "tanh"]}}')
        state_flags = state_flags or {}
        if not hasattr(self, "_eg_rsa_fired_event_rules"):
            self._eg_rsa_fired_event_rules = set()
        if not hasattr(self, "_eg_rsa_event_rule_duration_counts"):
            self._eg_rsa_event_rule_duration_counts = {}
        if not hasattr(self, "_eg_rsa_metric_stagnation_counts"):
            self._eg_rsa_metric_stagnation_counts = {}
        task_metrics = obs_map.get("task_metrics", {}) if isinstance(obs_map, dict) else {}
        prev_task_metrics = obs_map.get("_prev_task_metrics", {}) if isinstance(obs_map, dict) else {}
        def _get(name, default=0.0):
            if isinstance(obs_map, dict) and name in obs_map:
                return float(obs_map.get(name, default))
            if name in task_metrics:
                return float(task_metrics.get(name, default))
            return float(default)
        def _action_array():
            if action is None:
                return np.asarray([], dtype=float)
            return np.asarray(action, dtype=float).reshape(-1)
        def _map_action_primitives():
            metadata = schema.get("metadata", {}) or {}
            mapping = metadata.get("action_mapping", {}) or {}
            action_variables = metadata.get("action_variables", []) or []
            arr = _action_array()
            out = {}
            mapping_type = mapping.get("type")
            if mapping_type == "discrete_lookup":
                a = int(round(float(arr[0]))) if arr.size else 0
                for name, table in (mapping.get("variables", {}) or {}).items():
                    default = float(table.get("default", 0.0) or 0.0)
                    value = table.get(str(a), table.get(a, default))
                    out[str(name)] = float(value)
                return out
            for idx, value in enumerate(arr):
                out[f"action_{idx}"] = float(value)
            for idx, item in enumerate(action_variables):
                if isinstance(item, dict) and item.get("name"):
                    out[str(item["name"])] = float(arr[idx]) if idx < arr.size else 0.0
            return out
        def _vars():
            variables = {
                "x": _get("x", 0.0),
                "y": _get("y", 0.0),
                "vx": _get("vx", 0.0),
                "vy": _get("vy", 0.0),
                "angle": _get("angle", 0.0),
                "angular_velocity": _get("angular_velocity", _get("angularVelocity", 0.0)),
                "left_contact": bool(obs_map.get("left_contact", obs_map.get("leftContact", False))) if isinstance(obs_map, dict) else False,
                "right_contact": bool(obs_map.get("right_contact", obs_map.get("rightContact", False))) if isinstance(obs_map, dict) else False,
                "contact": bool(obs_map.get("contact", False)) if isinstance(obs_map, dict) else False,
                "both_contact": bool(obs_map.get("both_contact", False)) if isinstance(obs_map, dict) else False,
            }
            variables.update(_map_action_primitives())
            return variables
        def _eval_ast(node, variables):
            if isinstance(node, (int, float)):
                return float(node)
            if isinstance(node, bool):
                return bool(node)
            if not isinstance(node, dict):
                raise ValueError(f"Bad AST node: {node!r}")
            if "var" in node:
                return variables[str(node["var"])]
            if "const" in node:
                return float(node["const"])
            if "bool" in node:
                return bool(node["bool"])
            op = str(node.get("op", "")).lower()
            if op == "add":
                return sum(float(_eval_ast(x, variables)) for x in node["args"])
            if op == "mul":
                v = 1.0
                for x in node["args"]:
                    v *= float(_eval_ast(x, variables))
                return v
            if op == "sub":
                return float(_eval_ast(node["left"], variables)) - float(_eval_ast(node["right"], variables))
            if op == "div":
                d = float(_eval_ast(node["right"], variables))
                return 0.0 if abs(d) < 1e-12 else float(_eval_ast(node["left"], variables)) / d
            if op == "neg":
                return -float(_eval_ast(node["arg"], variables))
            if op == "abs":
                return abs(float(_eval_ast(node["arg"], variables)))
            if op == "sqrt":
                return math.sqrt(max(0.0, float(_eval_ast(node["arg"], variables))))
            if op == "exp":
                return math.exp(max(-50.0, min(50.0, float(_eval_ast(node["arg"], variables)))))
            if op == "tanh":
                return math.tanh(float(_eval_ast(node["arg"], variables)))
            if op == "min":
                return min(float(_eval_ast(x, variables)) for x in node["args"])
            if op == "max":
                return max(float(_eval_ast(x, variables)) for x in node["args"])
            if op == "clip":
                value = float(_eval_ast(node["args"][0], variables))
                low = float(_eval_ast(node["args"][1], variables))
                high = float(_eval_ast(node["args"][2], variables))
                if low > high:
                    low, high = high, low
                return max(low, min(high, value))
            if op == "and":
                return all(bool(_eval_ast(x, variables)) for x in node["args"])
            if op == "or":
                return any(bool(_eval_ast(x, variables)) for x in node["args"])
            if op == "not":
                return not bool(_eval_ast(node["arg"], variables))
            left = _eval_ast(node.get("left"), variables) if "left" in node else None
            right = _eval_ast(node.get("right"), variables) if "right" in node else None
            if op == "lt":
                return float(left) < float(right)
            if op == "le":
                return float(left) <= float(right)
            if op == "gt":
                return float(left) > float(right)
            if op == "ge":
                return float(left) >= float(right)
            if op == "eq":
                return left == right
            if op == "ne":
                return left != right
            raise ValueError(f"Unsupported AST op: {op}")
        def _clip(value, clip_range):
            if clip_range is None:
                return float(value)
            return float(np.clip(value, clip_range[0], clip_range[1]))
        variables = _vars()
        individual_reward = {}
        total_reward = 0.0
        for component in schema.get("components", []):
            if not component.get("enabled", True):
                continue
            name = component["name"]
            ctype = component["type"]
            params = component.get("params", {}) or {}
            weight = float(component.get("weight", 1.0))
            raw = 0.0
            if ctype == "formula_component":
                raw = float(_eval_ast(component.get("formula_ast") or params.get("formula_ast"), variables))
            elif ctype == "conditional_formula_component":
                cond = bool(_eval_ast(component.get("condition_ast") or params.get("condition_ast"), variables))
                raw = float(_eval_ast(component.get("formula_ast") or params.get("formula_ast"), variables)) if cond else 0.0
            elif ctype == "constant_alive":
                raw = float(params.get("value", 1.0))
            elif ctype == "event_bonus":
                raw = 1.0 if bool(state_flags.get(params.get("event", name), False)) else 0.0
            elif ctype == "metric_value":
                raw = float(task_metrics.get(params.get("metric", ""), 0.0))
            elif ctype == "metric_delta":
                metric = params.get("metric", "")
                current = float(task_metrics.get(metric, 0.0))
                previous = float(prev_task_metrics.get(metric, current))
                delta = current - previous
                raw = max(0.0, delta) if bool(params.get("positive_only", True)) else delta
            value = weight * _clip(raw, component.get("clip"))
            individual_reward[name] = float(value)
            total_reward += float(value)
        for rule in schema.get("event_rules", []):
            if not rule.get("enabled", True):
                continue
            name = rule["name"]
            condition = rule.get("condition", {}) or {}
            duration_steps = int(condition.get("duration_steps", 1) or 1)
            if rule.get("type") == "event_predicate":
                base_ok = bool(_eval_ast(condition.get("expr_ast") or rule.get("condition_ast"), variables))
            else:
                base_ok = True
                for key, expected in condition.items():
                    if key == "duration_steps":
                        continue
                    base_ok = base_ok and (bool(state_flags.get(key, False)) == bool(expected))
            if base_ok:
                self._eg_rsa_event_rule_duration_counts[name] = self._eg_rsa_event_rule_duration_counts.get(name, 0) + 1
            else:
                self._eg_rsa_event_rule_duration_counts[name] = 0
            ok = base_ok if duration_steps <= 1 else (base_ok and self._eg_rsa_event_rule_duration_counts.get(name, 0) >= duration_steps)
            if rule.get("one_time", False) and name in self._eg_rsa_fired_event_rules:
                value = 0.0
            else:
                value = float(rule.get("weight", 1.0)) if ok else 0.0
                if rule.get("one_time", False) and ok:
                    self._eg_rsa_fired_event_rules.add(name)
            individual_reward[name] = float(value)
            total_reward += float(value)
        individual_reward["reward"] = float(total_reward)
        return float(total_reward), individual_reward
