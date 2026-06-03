from __future__ import annotations

import json
from typing import Dict

from eg_rsa.reward.schema import RewardSchema


class RewardCompiler:
    """Compile a RewardSchema into stable-eureka compatible reward code.

    The generated code is intentionally conservative: it supports a small DSL of
    reward component types and writes every component into `individual_reward` so
    the existing stable-eureka evaluation callback can aggregate them.

    The target environment must expose the variables used by the compiled reward
    code at the point where `compute_reward(...)` is called.  For the first
    LunarLander implementation, the environment adapter should pass an `obs_map`,
    `action`, and `state_flags` dict into `compute_reward`.
    """

    SUPPORTED_COMPONENTS = {
        "distance_penalty",
        "distance_progress",
        "angle_penalty",
        "velocity_penalty",
        "action_penalty",
        "constant_alive",
        "event_bonus",
    }

    @staticmethod
    def compile(schema: RewardSchema, function_name: str = "compute_reward") -> str:
        RewardCompiler._validate(schema)
        schema_literal = json.dumps(schema.to_dict(), indent=4, ensure_ascii=False)
        return f'''# Generated code by EG-RSA
    def {function_name}(self, obs_map, action=None, state_flags=None):
        """Componentized reward generated from an EG-RSA RewardSchema.

        Parameters
        ----------
        obs_map : dict
            Semantic observation dictionary, e.g. {{"x": ..., "y": ...}}.
        action : Any
            Environment action used by action-cost components.
        state_flags : dict
            Boolean/event flags, e.g. {{"both_legs_contact": True}}.
        """
        import math
        import numpy as np

        schema = {schema_literal}
        state_flags = state_flags or {{}}
        individual_reward = {{}}
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
            params = component.get("params", {{}})
            raw = 0.0

            if ctype == "distance_penalty":
                # Negative Euclidean distance to a target point.
                target = params.get("target", [0.0 for _ in inputs])
                dist_sq = 0.0
                for idx, key in enumerate(inputs):
                    dist_sq += (_get(key) - float(target[idx])) ** 2
                raw = -math.sqrt(dist_sq)

            elif ctype == "distance_progress":
                # Positive progress is stored in obs_map by the environment adapter.
                progress_key = params.get("progress_key", "progress")
                raw = _get(progress_key)

            elif ctype == "angle_penalty":
                # Penalize absolute angular variables.
                raw = -sum(abs(_get(key)) for key in inputs)

            elif ctype == "velocity_penalty":
                # Penalize velocity norm.
                raw = -math.sqrt(sum(_get(key) ** 2 for key in inputs))

            elif ctype == "action_penalty":
                if action is None:
                    raw = 0.0
                else:
                    arr = np.asarray(action, dtype=float)
                    raw = -float(np.sum(np.square(arr)))

            elif ctype == "constant_alive":
                raw = float(params.get("value", 1.0))

            elif ctype == "event_bonus":
                event_name = params.get("event", name)
                raw = 1.0 if bool(state_flags.get(event_name, False)) else 0.0

            else:
                raw = 0.0

            value = weight * _clip(raw, component.get("clip"))
            individual_reward[name] = float(value)
            total_reward += float(value)

        # Event rules are compiled as safer, explicit event rewards.
        for rule in schema.get("event_rules", []):
            if not rule.get("enabled", True):
                continue
            name = rule["name"]
            weight = float(rule.get("weight", 1.0))
            condition = rule.get("condition", {{}})
            ok = True
            for key, expected in condition.items():
                if key == "duration_steps":
                    continue
                ok = ok and (state_flags.get(key, False) == expected)
            value = weight if ok else 0.0
            individual_reward[name] = float(value)
            total_reward += float(value)

        individual_reward["reward"] = float(total_reward)
        return float(total_reward), individual_reward
'''

    @staticmethod
    def _validate(schema: RewardSchema) -> None:
        names = set()
        for component in schema.components:
            if component.name in names:
                raise ValueError(f"Duplicate reward component name: {component.name}")
            names.add(component.name)
            if component.type not in RewardCompiler.SUPPORTED_COMPONENTS:
                raise ValueError(
                    f"Unsupported component type: {component.type}. "
                    f"Supported: {sorted(RewardCompiler.SUPPORTED_COMPONENTS)}"
                )
            if component.clip is not None:
                if len(component.clip) != 2 or component.clip[0] > component.clip[1]:
                    raise ValueError(f"Invalid clip range for {component.name}: {component.clip}")
