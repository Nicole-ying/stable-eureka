# Generated code by EG-RSA
    def compute_reward(self, obs_map, action=None, state_flags=None):
        import json
        import math
        import numpy as np
        schema = json.loads('{"version": 3, "components": [{"name": "r_progress", "type": "formula_component", "weight": 1.0, "inputs": [], "params": {"formula": "0"}, "clip": null, "enabled": true, "formula": "0"}, {"name": "r_stability", "type": "formula_component", "weight": 1.0, "inputs": [], "params": {"formula": "(-abs(angle) - 0.1*abs(angular_velocity))"}, "clip": null, "enabled": true, "formula": "(-abs(angle) - 0.1*abs(angular_velocity))"}, {"name": "r_control_cost", "type": "formula_component", "weight": 1.0, "inputs": [], "params": {"formula": "(-0.3*main_engine - 0.03*abs(side_engine))"}, "clip": null, "enabled": true, "formula": "(-0.3*main_engine - 0.03*abs(side_engine))"}], "event_rules": [{"name": "successful_landing", "type": "event_predicate", "weight": 1.0, "condition": {"expression": "left_contact and right_contact and abs(vx) < 0.2 and abs(vy) < 0.2 and abs(angle) < 0.1", "duration_steps": 1}, "one_time": true, "enabled": true}, {"name": "crash", "type": "event_predicate", "weight": 1.0, "condition": {"expression": "(y <= 0) and (not left_contact or not right_contact)", "duration_steps": 1}, "one_time": true, "enabled": true}], "metadata": {"source": "llm_bootstrap", "task": "LunarLander-v3", "reward_blueprint_present": true}}')
        state_flags = state_flags or {}
        if not hasattr(self, "_eg_rsa_fired_event_rules"):
            self._eg_rsa_fired_event_rules = set()
        if not hasattr(self, "_eg_rsa_event_rule_duration_counts"):
            self._eg_rsa_event_rule_duration_counts = {}
        if not hasattr(self, "_eg_rsa_metric_stagnation_counts"):
            self._eg_rsa_metric_stagnation_counts = {}
        task_metrics = {}
        prev_task_metrics = {}
        if isinstance(obs_map, dict):
            task_metrics = obs_map.get("task_metrics", {}) or {}
            prev_task_metrics = obs_map.get("_prev_task_metrics", {}) or {}
        if hasattr(self, "task_metrics"):
            task_metrics = getattr(self, "task_metrics") or task_metrics
        if hasattr(self, "_prev_task_metrics"):
            prev_task_metrics = getattr(self, "_prev_task_metrics") or prev_task_metrics
        individual_reward = {}
        total_reward = 0.0
        def _get(name, default=0.0):
            if isinstance(obs_map, dict) and name in obs_map:
                return float(obs_map.get(name, default))
            if name in task_metrics:
                return float(task_metrics.get(name, default))
            return float(default)
        def _action_value(index, default=0.0):
            if action is None:
                return float(default)
            arr = np.asarray(action, dtype=float).reshape(-1)
            return float(arr[index]) if index < len(arr) else float(default)
        def _clip_fn(value, low, high):
            return float(np.clip(float(value), float(low), float(high)))
        _allowed_formula_functions = {
            "abs": abs,
            "min": min,
            "max": max,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "tanh": math.tanh,
            "clip": _clip_fn,
        }
        def _primitive_vars():
            return {
                "x": _get("x", 0.0),
                "y": _get("y", 0.0),
                "vx": _get("vx", 0.0),
                "vy": _get("vy", 0.0),
                "angle": _get("angle", 0.0),
                "angular_velocity": _get("angular_velocity", _get("angularVelocity", 0.0)),
                "left_contact": bool(obs_map.get("left_contact", obs_map.get("leftContact", False))) if isinstance(obs_map, dict) else False,
                "right_contact": bool(obs_map.get("right_contact", obs_map.get("rightContact", False))) if isinstance(obs_map, dict) else False,
                "main_engine": _action_value(0, 0.0),
                "side_engine": _action_value(1, 0.0),
            }
        def _safe_formula(expr):
            if not isinstance(expr, str) or not expr.strip():
                return 0.0
            safe_locals = {}
            safe_locals.update(_allowed_formula_functions)
            safe_locals.update(_primitive_vars())
            return float(eval(compile(expr, "<eg_rsa_formula>", "eval"), {"__builtins__": {}}, safe_locals))
        def _clip(value, clip_range):
            if clip_range is None:
                return float(value)
            return float(np.clip(value, clip_range[0], clip_range[1]))
        for component in schema.get("components", []):
            if not component.get("enabled", True):
                continue
            name = component["name"]
            ctype = component["type"]
            weight = float(component.get("weight", 1.0))
            inputs = component.get("inputs", [])
            params = component.get("params", {})
            raw = 0.0
            if ctype == "distance_penalty":
                target = params.get("target", [0.0 for _ in inputs])
                raw = -math.sqrt(sum((_get(key) - float(target[idx])) ** 2 for idx, key in enumerate(inputs)))
            elif ctype == "distance_progress":
                raw = _get(params.get("progress_key", "progress"))
            elif ctype == "angle_penalty":
                raw = -sum(abs(_get(key)) for key in inputs)
            elif ctype == "velocity_penalty":
                raw = -math.sqrt(sum(_get(key) ** 2 for key in inputs))
            elif ctype == "action_penalty":
                raw = 0.0 if action is None else -float(np.sum(np.square(np.asarray(action, dtype=float))))
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
                if bool(params.get("positive_only", True)):
                    delta = max(0.0, delta)
                raw = float(delta)
            elif ctype == "metric_threshold_bonus":
                metric = params.get("metric", "")
                threshold = float(params.get("threshold", 0.0))
                direction = params.get("direction", "ge")
                value = float(task_metrics.get(metric, 0.0))
                if direction == "le":
                    raw = 1.0 if value <= threshold else 0.0
                else:
                    raw = 1.0 if value >= threshold else 0.0
            elif ctype == "metric_stagnation_penalty":
                metric = params.get("metric", "")
                threshold = float(params.get("threshold", 1e-3))
                window = int(params.get("window", 20))
                current = float(task_metrics.get(metric, 0.0))
                previous = float(prev_task_metrics.get(metric, current))
                delta = abs(current - previous)
                if delta < threshold:
                    self._eg_rsa_metric_stagnation_counts[name] = self._eg_rsa_metric_stagnation_counts.get(name, 0) + 1
                else:
                    self._eg_rsa_metric_stagnation_counts[name] = 0
                raw = -1.0 if self._eg_rsa_metric_stagnation_counts.get(name, 0) >= window else 0.0
            elif ctype == "formula_component":
                raw = _safe_formula(component.get("formula") or params.get("formula", "0.0"))
            elif ctype == "conditional_formula_component":
                cond_expr = component.get("condition") or params.get("condition", "False")
                raw = _safe_formula(component.get("formula") or params.get("formula", "0.0")) if bool(_safe_formula(cond_expr)) else 0.0
            value = weight * _clip(raw, component.get("clip"))
            individual_reward[name] = float(value)
            total_reward += float(value)
        for rule in schema.get("event_rules", []):
            if not rule.get("enabled", True):
                continue
            name = rule["name"]
            weight = float(rule.get("weight", 1.0))
            rtype = rule.get("type", "event_bonus")
            condition = rule.get("condition", {})
            duration_steps = int(condition.get("duration_steps", 1) or 1) if isinstance(condition, dict) else 1
            if rtype == "event_predicate":
                expr = condition.get("expression") or condition.get("formula") or "False"
                base_ok = bool(_safe_formula(expr))
            else:
                base_ok = True
                for key, expected in condition.items():
                    if key == "duration_steps":
                        continue
                    base_ok = base_ok and (state_flags.get(key, False) == expected)
            if base_ok:
                self._eg_rsa_event_rule_duration_counts[name] = self._eg_rsa_event_rule_duration_counts.get(name, 0) + 1
            else:
                self._eg_rsa_event_rule_duration_counts[name] = 0
            ok = base_ok if duration_steps <= 1 else (base_ok and self._eg_rsa_event_rule_duration_counts.get(name, 0) >= duration_steps)
            if rule.get("one_time", False) and name in self._eg_rsa_fired_event_rules:
                value = 0.0
            else:
                value = float(weight) if ok else 0.0
                if rule.get("one_time", False) and ok:
                    self._eg_rsa_fired_event_rules.add(name)
            individual_reward[name] = float(value)
            total_reward += float(value)
        individual_reward["reward"] = float(total_reward)
        return float(total_reward), individual_reward
