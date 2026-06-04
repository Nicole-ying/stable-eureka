# Generated code by EG-RSA
    def compute_reward(self, obs_map, action=None, state_flags=None):
        import json
        import math
        import numpy as np
        schema = json.loads('{"version": 1, "components": [{"name": "r_distance", "type": "distance_penalty", "weight": 1.0, "inputs": ["x", "y"], "params": {"target": [0.0, 0.0]}, "clip": [-2.0, 0.0], "enabled": true}, {"name": "r_velocity", "type": "velocity_penalty", "weight": 0.5, "inputs": ["vx", "vy"], "params": {}, "clip": [-2.0, 0.0], "enabled": true}, {"name": "r_angle", "type": "angle_penalty", "weight": 0.3, "inputs": ["angle", "angular_velocity"], "params": {}, "clip": [-2.0, 0.0], "enabled": true}, {"name": "r_energy", "type": "action_penalty", "weight": 0.03, "inputs": ["action"], "params": {}, "clip": [-1.0, 0.0], "enabled": true}, {"name": "r_contact", "type": "event_bonus", "weight": 20.0, "inputs": [], "params": {"event": "contact"}, "clip": [-1.0, 1.0], "enabled": true}], "event_rules": [], "metadata": {"env": "LunarLander-v3", "note": "Initial hand-written schema for validating the EG-RSA minimal loop. Later versions should be edited through operators."}}')
        state_flags = state_flags or {}
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
            value = weight * _clip(raw, component.get("clip"))
            individual_reward[name] = float(value)
            total_reward += float(value)
        for rule in schema.get("event_rules", []):
            if not rule.get("enabled", True):
                continue
            condition = rule.get("condition", {})
            ok = True
            for key, expected in condition.items():
                if key == "duration_steps":
                    continue
                ok = ok and (state_flags.get(key, False) == expected)
            value = float(rule.get("weight", 1.0)) if ok else 0.0
            individual_reward[rule["name"]] = value
            total_reward += value
        individual_reward["reward"] = float(total_reward)
        return float(total_reward), individual_reward
