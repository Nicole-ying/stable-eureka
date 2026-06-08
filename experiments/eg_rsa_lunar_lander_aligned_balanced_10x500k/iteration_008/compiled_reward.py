# Generated code by EG-RSA
    def compute_reward(self, obs_map, action=None, state_flags=None):
        import json
        import math
        import numpy as np
        schema = json.loads('{"version": 2, "components": [{"name": "r_approach_region", "type": "metric_value", "weight": 0.705, "inputs": [], "params": {"metric": "approach_region_score"}, "clip": [0.0, 1.0], "enabled": true}, {"name": "r_approach_progress", "type": "metric_delta", "weight": 5.0, "inputs": [], "params": {"metric": "approach_region_score", "positive_only": true}, "clip": [0.0, 1.0], "enabled": true}, {"name": "r_stability", "type": "metric_value", "weight": 1.0, "inputs": [], "params": {"metric": "stability"}, "clip": [0.0, 1.0], "enabled": true}, {"name": "r_landing_quality", "type": "metric_value", "weight": 2.0, "inputs": [], "params": {"metric": "landing_quality"}, "clip": [0.0, 1.0], "enabled": true}, {"name": "r_energy", "type": "action_penalty", "weight": 0.02, "inputs": ["action"], "params": {}, "clip": [-1.0, 0.0], "enabled": true}], "event_rules": [{"name": "r_safe_contact_once", "type": "event_bonus", "weight": 10.0, "condition": {"safe_contact": true, "duration_steps": 2}, "one_time": true, "enabled": true}, {"name": "r_stable_landing_once", "type": "event_bonus", "weight": 80.0, "condition": {"stable_landing_condition": true, "duration_steps": 3}, "one_time": true, "enabled": true}], "metadata": {"env": "LunarLander-v3", "note": "Aligned initial schema for EG-RSA. It rewards safe-region approach and stable landing process rather than raw contact or point-distance only. Later versions should still be edited through operators."}}')
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
            return float(obs_map.get(name, default))
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
            value = weight * _clip(raw, component.get("clip"))
            individual_reward[name] = float(value)
            total_reward += float(value)
        for rule in schema.get("event_rules", []):
            if not rule.get("enabled", True):
                continue
            name = rule["name"]
            weight = float(rule.get("weight", 1.0))
            condition = rule.get("condition", {})
            duration_steps = int(condition.get("duration_steps", 1) or 1)
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
