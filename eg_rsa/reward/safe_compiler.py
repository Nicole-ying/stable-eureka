from __future__ import annotations

import json

from eg_rsa.reward.schema import RewardSchema


class SafeRewardCompiler:
    """Safer compact compiler for EG-RSA reward schemas.

    This version embeds the schema through json.loads(repr(json_text)) so the
    generated Python code never contains raw JSON booleans such as true/null.
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
{i}    individual_reward = {{}}
{i}    total_reward = 0.0
{i}    def _get(name, default=0.0):
{i}        return float(obs_map.get(name, default))
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
{i}            raw = 0.0 if action is None else -float(np.sum(np.square(np.asarray(action, dtype=float))))
{i}        elif ctype == "constant_alive":
{i}            raw = float(params.get("value", 1.0))
{i}        elif ctype == "event_bonus":
{i}            raw = 1.0 if bool(state_flags.get(params.get("event", name), False)) else 0.0
{i}        value = weight * _clip(raw, component.get("clip"))
{i}        individual_reward[name] = float(value)
{i}        total_reward += float(value)
{i}    for rule in schema.get("event_rules", []):
{i}        if not rule.get("enabled", True):
{i}            continue
{i}        condition = rule.get("condition", {{}})
{i}        ok = True
{i}        for key, expected in condition.items():
{i}            if key == "duration_steps":
{i}                continue
{i}            ok = ok and (state_flags.get(key, False) == expected)
{i}        value = float(rule.get("weight", 1.0)) if ok else 0.0
{i}        individual_reward[rule["name"]] = value
{i}        total_reward += value
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
                raise ValueError(f"Unsupported component type: {component.type}")
            if component.clip is not None:
                if len(component.clip) != 2 or component.clip[0] > component.clip[1]:
                    raise ValueError(f"Invalid clip range for {component.name}: {component.clip}")
