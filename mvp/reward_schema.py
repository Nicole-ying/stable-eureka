from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import types
from typing import Any

import numpy as np


FORBIDDEN_NAMES = {
    "env_reward",
    "fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
    "_hidden_env_reward",
    "compute_fitness_score",
}

FORBIDDEN_SYNTAX = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.Try,
    ast.ClassDef,
    ast.Lambda,
    ast.Global,
    ast.Nonlocal,
)

REQUIRED_SIGNATURE = ["obs", "action", "next_obs", "done", "info"]


def build_default_schema(clean_interface: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "env_alias": clean_interface.get("env_alias"),
        "task_description": clean_interface.get("eureka_task_description", "")[:2000],
        "step_code": clean_interface.get("eureka_step_code", "")[:2000],
    }
    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

    return {
        "schema_version": f"eg_rsa_reward_schema_v1_{schema_hash}",
        "env_alias": clean_interface.get("env_alias", "Env-unknown"),
        "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
        "return_contract": "return float(total_reward), components_dict",
        "allowed_inputs": REQUIRED_SIGNATURE,
        "private_signal_policy": "Generated reward code must not use env_reward, fitness_score, or hidden evaluator details.",
        "components": [
            {
                "id": "progress",
                "description": "dense task-progress shaping inferred from public task context",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "stability",
                "description": "bounded shaping for stable/safe task-relevant behavior inferred from public task context",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "effort",
                "description": "bounded penalty for unnecessary or costly actions",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "terminal",
                "description": "bounded terminal shaping from public done signal",
                "direction": "maximize",
                "required": True,
            },
        ],
        "reward_abs_bound": 1000.0,
    }


def normalize_schema(raw: dict[str, Any] | None, clean_interface: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize an LLM-generated reward schema.

    Quality-gate v1 policy:
      - If the LLM provides at least 4 valid components, trust that schema.
      - Do not blindly append default progress/stability/effort/terminal.
      - Keep required component count between 4 and 6 where possible.
      - Avoid multiple terminal-like components activating on done.
      - Fall back to default schema only when LLM schema is missing or too small.
    """
    if not isinstance(raw, dict):
        raw = {}

    default = build_default_schema(clean_interface)

    schema = dict(default)
    schema.update({k: v for k, v in raw.items() if v is not None})

    raw_components = raw.get("components")
    if not isinstance(raw_components, list):
        raw_components = []

    normalized_components = []
    seen = set()
    terminal_like_seen = False

    for c in raw_components:
        if not isinstance(c, dict):
            continue

        cid = str(c.get("id", "")).strip()
        if not cid:
            continue

        cid_norm = cid.lower()
        is_terminal_like = any(
            term in cid_norm
            for term in ("terminal", "landing", "crash", "success", "failure", "done")
        )

        # Keep one terminal-outcome component only. This prevents schemas such as
        # landing_bonus + crash_penalty + terminal all being required simultaneously.
        if is_terminal_like:
            if terminal_like_seen:
                continue
            cid = "terminal"
            terminal_like_seen = True

        if cid in seen:
            continue

        seen.add(cid)
        normalized_components.append(
            {
                "id": cid,
                "description": str(c.get("description", f"{cid} component")),
                "direction": str(c.get("direction", "maximize")),
                "required": bool(c.get("required", True)),
            }
        )

    # If LLM schema is too small or malformed, use default compact schema.
    if len(normalized_components) < 4:
        normalized_components = list(default["components"])
    else:
        # Cap at 6 required components to keep reward code compact and interpretable.
        required = [c for c in normalized_components if c.get("required", True)]
        optional = [c for c in normalized_components if not c.get("required", True)]

        if len(required) > 6:
            kept_required = required[:6]
            kept_ids = {c["id"] for c in kept_required}
            normalized_components = kept_required + [c for c in optional if c["id"] in kept_ids]
        else:
            normalized_components = normalized_components[:6]

    schema["components"] = normalized_components
    schema["reward_signature"] = "compute_reward(obs, action, next_obs, done, info)"
    schema["return_contract"] = "return float(total_reward), components_dict"
    schema["allowed_inputs"] = REQUIRED_SIGNATURE
    schema["reward_abs_bound"] = float(schema.get("reward_abs_bound", 1000.0))

    payload = {
        "env_alias": clean_interface.get("env_alias"),
        "components": schema["components"],
        "task_head": clean_interface.get("eureka_task_description", "")[:1000],
    }
    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

    # Quality-gate v1.1:
    # Schema version must reflect the final normalized component set.
    # Do not preserve a default schema_version after LLM components are normalized,
    # otherwise different schemas can share the same version and contaminate memory retrieval.
    schema["schema_version"] = f"eg_rsa_reward_schema_v1_{schema_hash}"

    return schema


def _sample_obs(clean_interface: dict[str, Any]):
    if clean_interface.get("interface_mode") == "anonymous_clean":
        space = clean_interface.get("observation_space", {})
        if space.get("type") == "Box":
            shape = tuple(space.get("shape", [64]))
            return np.zeros(shape, dtype=np.float32)
    # Eureka step code commonly uses vector observations. Use a long generic vector for smoke test.
    return np.zeros((64,), dtype=np.float32)


def _sample_action(clean_interface: dict[str, Any]):
    return 0


def _find_compute_reward(tree: ast.AST) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
            return node
    return None


def _compile_reward(reward_code: str):
    tree = ast.parse(reward_code, mode="exec")
    module = types.ModuleType("reward_module")
    module.__dict__["np"] = np
    module.__dict__["math"] = math
    exec(compile(tree, filename="<reward_code>", mode="exec"), module.__dict__)
    return module.__dict__.get("compute_reward")


def validate_reward_code(
    reward_code: str,
    schema: dict[str, Any],
    clean_interface: dict[str, Any],
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    raw_lower = reward_code.lower()
    for name in FORBIDDEN_NAMES:
        if name.lower() in raw_lower:
            errors.append(f"forbidden token appears in code: {name}")

    try:
        tree = ast.parse(reward_code, mode="exec")
    except SyntaxError as e:
        return False, [f"syntax_error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_SYNTAX):
            errors.append(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            errors.append(f"forbidden attribute: {node.attr}")

    fn_node = _find_compute_reward(tree)
    if fn_node is None:
        errors.append("missing function: compute_reward")
    else:
        args = [a.arg for a in fn_node.args.args]
        if args != REQUIRED_SIGNATURE:
            errors.append(f"bad signature: expected {REQUIRED_SIGNATURE}, got {args}")

    if errors:
        return False, sorted(set(errors))

    try:
        fn = _compile_reward(reward_code)
        if fn is None:
            return False, ["compute_reward not found after compilation"]

        obs = _sample_obs(clean_interface)
        next_obs = copy.deepcopy(obs)
        action = _sample_action(clean_interface)

        out = fn(obs, action, next_obs, False, {})
        if not isinstance(out, tuple) or len(out) != 2:
            errors.append("compute_reward must return (total_reward, components_dict)")
        else:
            total, components = out
            total = float(total)
            if not np.isfinite(total):
                errors.append("total reward is not finite")
            if abs(total) > float(schema.get("reward_abs_bound", 1000.0)):
                errors.append("total reward exceeds reward_abs_bound")
            if not isinstance(components, dict):
                errors.append("components must be a dict")
            else:
                schema_ids = [c["id"] for c in schema.get("components", [])]
                schema_id_set = set(schema_ids)
                required_ids = [
                    c["id"]
                    for c in schema.get("components", [])
                    if c.get("required", False)
                ]
                missing = [cid for cid in required_ids if cid not in components]
                if missing:
                    errors.append(f"missing required components: {missing}")

                extra = [cid for cid in components if cid not in schema_id_set]
                if extra:
                    errors.append(f"extra components not in reward schema: {extra}")

                # Keep exact schema order for compiled RewardSpec code and stable diagnostics.
                if list(components.keys()) != [cid for cid in schema_ids if cid in components]:
                    errors.append("components keys must follow reward schema order")

                for k, v in components.items():
                    try:
                        fv = float(v)
                    except Exception:
                        errors.append(f"component {k} is not float-convertible")
                        continue
                    if not np.isfinite(fv):
                        errors.append(f"component {k} is not finite")

    except Exception as e:
        errors.append(f"smoke_test_error: {type(e).__name__}: {e}")

    return len(errors) == 0, sorted(set(errors))
