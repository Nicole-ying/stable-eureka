from __future__ import annotations

import ast
import hashlib
import json
from typing import Any


PRIVATE_TERMS = (
    "env_reward",
    "hidden_env_reward",
    "_hidden_env_reward",
    "fitness_score",
    "compute_fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
)

ALLOWED_NAMES = {
    "obs",
    "next_obs",
    "action",
    "done",
    "info",
    "np",
    "math",
    "abs",
    "min",
    "max",
    "float",
    "int",
    "bool",
    "True",
    "False",
}

ALLOWED_MATH_ATTRS = {
    "sqrt",
    "exp",
    "tanh",
    "sin",
    "cos",
    "log",
}

ALLOWED_NP_ATTRS = {
    "sqrt",
    "exp",
    "tanh",
    "sin",
    "cos",
    "log",
    "clip",
    "minimum",
    "maximum",
}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.IfExp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Attribute,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


class RewardSpecError(ValueError):
    pass


def stable_spec_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _contains_private_term(text: str) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in PRIVATE_TERMS)


def _schema_component_ids(schema: dict[str, Any]) -> list[str]:
    return [str(c.get("id", "")).strip() for c in schema.get("components", []) if str(c.get("id", "")).strip()]


def _required_component_ids(schema: dict[str, Any]) -> list[str]:
    return [
        str(c.get("id", "")).strip()
        for c in schema.get("components", [])
        if c.get("required", True) and str(c.get("id", "")).strip()
    ]


def _validate_clip(value: Any, field_name: str) -> tuple[float, float]:
    if value is None:
        return (-1000.0, 1000.0)
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise RewardSpecError(f"{field_name} must be [low, high]")
    lo = float(value[0])
    hi = float(value[1])
    if lo > hi:
        raise RewardSpecError(f"{field_name} low > high")
    return lo, hi


def _validate_expression(expr: str, *, component_id: str) -> None:
    if not isinstance(expr, str) or not expr.strip():
        raise RewardSpecError(f"component {component_id}: empty expression")
    if _contains_private_term(expr):
        raise RewardSpecError(f"component {component_id}: private term appears in expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RewardSpecError(f"component {component_id}: expression syntax error: {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise RewardSpecError(f"component {component_id}: unsupported expression node {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES:
                raise RewardSpecError(f"component {component_id}: unsupported name {node.id}")

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "math":
                if node.attr not in ALLOWED_MATH_ATTRS:
                    raise RewardSpecError(f"component {component_id}: unsupported math.{node.attr}")
            elif isinstance(node.value, ast.Name) and node.value.id == "np":
                if node.attr not in ALLOWED_NP_ATTRS:
                    raise RewardSpecError(f"component {component_id}: unsupported np.{node.attr}")
            elif isinstance(node.value, ast.Name) and node.value.id == "info" and node.attr == "get":
                pass
            else:
                raise RewardSpecError(f"component {component_id}: unsupported attribute access")

        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                if fn.id not in {"abs", "min", "max", "float", "int", "bool"}:
                    raise RewardSpecError(f"component {component_id}: unsupported function {fn.id}")
            elif isinstance(fn, ast.Attribute):
                if isinstance(fn.value, ast.Name) and fn.value.id == "math" and fn.attr in ALLOWED_MATH_ATTRS:
                    pass
                elif isinstance(fn.value, ast.Name) and fn.value.id == "np" and fn.attr in ALLOWED_NP_ATTRS:
                    pass
                elif isinstance(fn.value, ast.Name) and fn.value.id == "info" and fn.attr == "get":
                    for arg in node.args[:1]:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if _contains_private_term(arg.value):
                                raise RewardSpecError(f"component {component_id}: private info key")
                        else:
                            raise RewardSpecError(f"component {component_id}: info.get key must be literal string")
                else:
                    raise RewardSpecError(f"component {component_id}: unsupported function attribute")
            else:
                raise RewardSpecError(f"component {component_id}: unsupported call target")


def normalize_reward_spec(raw_spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_spec, dict):
        raise RewardSpecError("reward spec must be a JSON object")

    raw_components = raw_spec.get("components")
    if not isinstance(raw_components, list):
        raise RewardSpecError("reward spec must contain components list")

    schema_ids = _schema_component_ids(schema)
    required_ids = _required_component_ids(schema)
    allowed_set = set(schema_ids)
    required_set = set(required_ids)

    by_id: dict[str, dict[str, Any]] = {}
    for c in raw_components:
        if not isinstance(c, dict):
            raise RewardSpecError("each component spec must be an object")
        cid = str(c.get("id", "")).strip()
        if not cid:
            raise RewardSpecError("component spec missing id")
        if cid not in allowed_set:
            raise RewardSpecError(f"component {cid} is not in reward schema")
        if cid in by_id:
            raise RewardSpecError(f"duplicate component id: {cid}")

        expr = str(c.get("expression", "")).strip()
        _validate_expression(expr, component_id=cid)
        clip = _validate_clip(c.get("clip", [-1000.0, 1000.0]), f"component {cid}.clip")

        by_id[cid] = {
            "id": cid,
            "expression": expr,
            "clip": [clip[0], clip[1]],
            "description": str(c.get("description", "")).strip(),
        }

    missing = [cid for cid in required_ids if cid not in by_id]
    if missing:
        raise RewardSpecError(f"missing required components: {missing}")

    extra = [cid for cid in by_id if cid not in allowed_set]
    if extra:
        raise RewardSpecError(f"extra components not in schema: {extra}")

    # Keep component order identical to the schema. This makes diffs and memory stable.
    ordered = [by_id[cid] for cid in schema_ids if cid in by_id]

    final_clip = _validate_clip(raw_spec.get("final_clip", [-1000.0, 1000.0]), "final_clip")
    total = str(raw_spec.get("total", "sum_components")).strip() or "sum_components"
    if total != "sum_components":
        raise RewardSpecError("reward_spec.total must be 'sum_components' in v1")

    spec = {
        "reward_spec_version": "eg_rsa_reward_spec_v1",
        "schema_version": schema.get("schema_version"),
        "rationale": str(raw_spec.get("rationale", "")).strip(),
        "components": ordered,
        "total": "sum_components",
        "final_clip": [final_clip[0], final_clip[1]],
    }
    spec["spec_id"] = f"reward_spec_{stable_spec_hash(spec)}"
    return spec


def compile_reward_spec_to_code(spec: dict[str, Any]) -> str:
    components = spec.get("components", [])
    if not components:
        raise RewardSpecError("cannot compile empty reward spec")

    lines: list[str] = []
    lines.append("def compute_reward(obs, action, next_obs, done, info):")
    lines.append("    components = {}")
    total_terms: list[str] = []

    for idx, c in enumerate(components):
        cid = str(c["id"])
        var = f"component_{idx}"
        expr = str(c["expression"])
        lo, hi = _validate_clip(c.get("clip", [-1000.0, 1000.0]), f"component {cid}.clip")
        lines.append(f"    # component: {cid}")
        lines.append(f"    {var} = float({expr})")
        lines.append(f"    {var} = max(min({var}, {hi!r}), {lo!r})")
        lines.append(f"    components[{cid!r}] = float({var})")
        total_terms.append(var)

    flo, fhi = _validate_clip(spec.get("final_clip", [-1000.0, 1000.0]), "final_clip")
    lines.append(f"    total_reward = {' + '.join(total_terms)}")
    lines.append(f"    total_reward = max(min(total_reward, {fhi!r}), {flo!r})")
    lines.append("    return float(total_reward), components")
    lines.append("")
    return "\n".join(lines)


def parse_validate_compile_reward_spec(raw_spec: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], str]:
    spec = normalize_reward_spec(raw_spec, schema)
    code = compile_reward_spec_to_code(spec)
    return spec, code
