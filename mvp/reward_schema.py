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


def build_default_schema(public_task: dict[str, Any], clean_interface: dict[str, Any]) -> dict[str, Any]:
    """
    BootstrapAgent 生成初始 schema。

    这里故意用通用组件，不写任务专属物理语义：
      progress  : 泛化的任务进展代理；
      stability : 状态变化/姿态/数值稳定性代理；
      effort    : 动作幅值或动作切换成本；
      terminal  : done 时的终端 shaping。
    """
    payload = {
        "env_alias": clean_interface["env_alias"],
        "observation_space": clean_interface["observation_space"],
        "action_space": clean_interface["action_space"],
        "task_goal": public_task["task_goal"],
        "task_style": public_task["task_style"],
    }
    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

    return {
        "schema_version": f"clean_reward_schema_v1_{schema_hash}",
        "env_alias": clean_interface["env_alias"],
        "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
        "return_contract": "return float(total_reward), components_dict",
        "allowed_inputs": REQUIRED_SIGNATURE,
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "components": [
            {
                "id": "progress",
                "description": "dense task progress proxy inferred only from obs/action transitions",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "stability",
                "description": "bounded penalty for unstable or abrupt transitions",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "effort",
                "description": "bounded penalty for unnecessary action magnitude or switching",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "terminal",
                "description": "bounded terminal success/failure shaping without reading hidden reward",
                "direction": "maximize",
                "required": True,
            },
        ],
        "reward_abs_bound": 1000.0,
    }


def _sample_obs(space_dict: dict[str, Any]):
    if space_dict.get("type") == "Box":
        shape = tuple(space_dict.get("shape", []))
        return np.zeros(shape, dtype=np.float32)
    if space_dict.get("type") == "Discrete":
        return int(space_dict.get("start", 0))
    if space_dict.get("type") == "MultiDiscrete":
        return np.zeros(len(space_dict.get("nvec", [])), dtype=np.int64)
    if space_dict.get("type") == "MultiBinary":
        return np.zeros(space_dict.get("n", 1), dtype=np.int8)
    return np.zeros((1,), dtype=np.float32)


def _sample_action(space_dict: dict[str, Any]):
    if space_dict.get("type") == "Box":
        shape = tuple(space_dict.get("shape", []))
        return np.zeros(shape, dtype=np.float32)
    if space_dict.get("type") == "Discrete":
        return int(space_dict.get("start", 0))
    if space_dict.get("type") == "MultiDiscrete":
        return np.zeros(len(space_dict.get("nvec", [])), dtype=np.int64)
    if space_dict.get("type") == "MultiBinary":
        return np.zeros(space_dict.get("n", 1), dtype=np.int8)
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

        obs = _sample_obs(clean_interface["observation_space"])
        next_obs = copy.deepcopy(obs)
        action = _sample_action(clean_interface["action_space"])

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
                required_ids = [
                    c["id"]
                    for c in schema.get("components", [])
                    if c.get("required", False)
                ]
                missing = [cid for cid in required_ids if cid not in components]
                if missing:
                    errors.append(f"missing required components: {missing}")
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
