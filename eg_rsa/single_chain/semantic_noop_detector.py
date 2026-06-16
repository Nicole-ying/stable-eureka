from __future__ import annotations

import ast
import difflib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


@dataclass
class ProbeCase:
    state: List[float]
    m_power: float
    s_power: float
    terminated: bool
    episode_length: int


DEFAULT_PROBES = [
    ProbeCase([0.0, 1.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0, False, 25),
    ProbeCase([0.2, 0.5, -0.1, -0.3, 0.15, 0.02, 0.0, 0.0], 0.7, 0.0, False, 120),
    ProbeCase([0.03, 0.05, 0.01, -0.05, 0.02, 0.0, 1.0, 1.0], 0.0, 0.0, False, 420),
    ProbeCase([0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 0.0, 0.0, True, 280),
    ProbeCase([0.6, 0.1, 0.3, -0.6, 0.6, 0.1, 0.0, 0.0], 1.0, 0.5, True, 80),
    ProbeCase([-0.15, 0.2, 0.05, -0.1, -0.2, 0.03, 1.0, 0.0], 0.2, 0.7, False, 760),
]

ALLOWED_CALL_ROOTS = {"abs", "float", "int", "min", "max", "bool", "hasattr"}
ALLOWED_MODULE_CALLS = {"math", "np", "numpy"}
DISALLOWED_NODES = (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Raise, ast.Lambda, ast.Global, ast.Nonlocal)


def detect_semantic_noop_edit(
    old_code: str | None,
    new_code: str | None,
    probes: List[ProbeCase] | None = None,
    reward_abs_tol: float = 1e-6,
    component_abs_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Detect whether a revised reward is effectively the same as its parent.

    This is a training-cost guard. It does not prove two programs are equivalent;
    it catches common cosmetic edits such as variable reordering, comments, or
    algebraically identical reward sums by comparing normalized code and reward
    outputs on expert-designed probe states.
    """
    old_code = old_code or ""
    new_code = new_code or ""
    if not old_code.strip() or not new_code.strip():
        return _skip_report("missing_reference_or_candidate_code")

    old_probe_safe = _probe_safe(old_code)
    new_probe_safe = _probe_safe(new_code)
    if not old_probe_safe.get("safe") or not new_probe_safe.get("safe"):
        return {
            **_skip_report("reward code is outside conservative probe-safe subset"),
            "old_probe_safety": old_probe_safe,
            "new_probe_safety": new_probe_safe,
        }

    normalized_old = _normalize_ast(old_code)
    normalized_new = _normalize_ast(new_code)
    source_similarity = difflib.SequenceMatcher(None, _strip_ws(old_code), _strip_ws(new_code)).ratio()
    ast_equal = bool(normalized_old and normalized_new and normalized_old == normalized_new)

    probe_report = _probe_equivalence(old_code, new_code, probes or DEFAULT_PROBES, reward_abs_tol, component_abs_tol)
    outputs_equivalent = bool(probe_report.get("outputs_equivalent"))

    semantic_noop = bool(ast_equal or (source_similarity >= 0.985 and outputs_equivalent) or (source_similarity >= 0.94 and outputs_equivalent and probe_report.get("component_keys_equivalent")))
    reason = "semantic_change_detected"
    if semantic_noop:
        reason = "candidate reward is semantically equivalent to parent on AST/source/probe checks"
    elif outputs_equivalent:
        reason = "probe outputs equivalent but source changed enough to allow training"

    return {
        "file_type": "semantic_noop_report",
        "valid": not semantic_noop,
        "semantic_noop_edit": semantic_noop,
        "should_train": not semantic_noop,
        "reason": reason,
        "source_similarity": source_similarity,
        "ast_equal": ast_equal,
        "probe_equivalence": probe_report,
        "hard_constraints": [
            "Do not spend PPO training budget on semantic no-op reward edits.",
            "If blocked, regenerate a structurally meaningful reward change that alters expected target behavior.",
        ] if semantic_noop else [],
    }


def _skip_report(reason: str) -> Dict[str, Any]:
    return {
        "file_type": "semantic_noop_report",
        "valid": True,
        "semantic_noop_edit": False,
        "should_train": True,
        "stage": "skipped",
        "reason": reason,
    }


def _probe_safe(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except Exception as exc:
        return {"safe": False, "reason": f"parse_error: {exc}"}
    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_NODES):
            return {"safe": False, "reason": f"disallowed_ast_node:{type(node).__name__}"}
        if isinstance(node, ast.Call) and not _allowed_call(node):
            return {"safe": False, "reason": "disallowed_call"}
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id.startswith("__"):
            return {"safe": False, "reason": "dunder_attribute"}
    return {"safe": True, "reason": "probe_safe_subset"}


def _allowed_call(node: ast.Call) -> bool:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id in ALLOWED_CALL_ROOTS
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        return fn.value.id in ALLOWED_MODULE_CALLS
    return False


def _normalize_ast(code: str) -> str:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                if hasattr(node, attr):
                    setattr(node, attr, None)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        return ""


def _strip_ws(code: str) -> str:
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _probe_equivalence(
    old_code: str,
    new_code: str,
    probes: List[ProbeCase],
    reward_abs_tol: float,
    component_abs_tol: float,
) -> Dict[str, Any]:
    try:
        old_fn = _load_reward_fn(old_code)
        new_fn = _load_reward_fn(new_code)
    except Exception as exc:
        return {
            "outputs_equivalent": False,
            "component_keys_equivalent": False,
            "probe_error": str(exc),
            "num_probes": len(probes),
        }

    max_reward_delta = 0.0
    max_component_delta = 0.0
    mismatches: List[Dict[str, Any]] = []
    component_keys_equal = True

    for i, probe in enumerate(probes):
        old_obj = _DummyRewardContext(probe.episode_length)
        new_obj = _DummyRewardContext(probe.episode_length)
        try:
            old_reward, old_info = old_fn(old_obj, list(probe.state), probe.m_power, probe.s_power, probe.terminated)
            new_reward, new_info = new_fn(new_obj, list(probe.state), probe.m_power, probe.s_power, probe.terminated)
        except Exception as exc:
            return {
                "outputs_equivalent": False,
                "component_keys_equivalent": False,
                "probe_error": f"probe_{i}: {exc}",
                "num_probes": len(probes),
            }
        reward_delta = abs(_num(old_reward) - _num(new_reward))
        max_reward_delta = max(max_reward_delta, reward_delta)
        old_info = old_info if isinstance(old_info, dict) else {}
        new_info = new_info if isinstance(new_info, dict) else {}
        if set(old_info.keys()) != set(new_info.keys()):
            component_keys_equal = False
        for key in sorted(set(old_info.keys()) | set(new_info.keys())):
            component_delta = abs(_num(old_info.get(key)) - _num(new_info.get(key)))
            max_component_delta = max(max_component_delta, component_delta)
        if reward_delta > reward_abs_tol:
            mismatches.append({"probe_index": i, "reward_delta": reward_delta})

    outputs_equivalent = bool(max_reward_delta <= reward_abs_tol and max_component_delta <= component_abs_tol)
    return {
        "outputs_equivalent": outputs_equivalent,
        "component_keys_equivalent": component_keys_equal,
        "num_probes": len(probes),
        "max_reward_delta": max_reward_delta,
        "max_component_delta": max_component_delta,
        "mismatches": mismatches[:5],
    }


def _load_reward_fn(code: str) -> Callable[..., Tuple[float, Dict[str, Any]]]:
    namespace: Dict[str, Any] = {"math": math, "np": np, "numpy": np, "__builtins__": {}}
    exec(compile(code, "<reward_code>", "exec"), namespace)
    fn = namespace.get("compute_reward")
    if not callable(fn):
        raise ValueError("compute_reward is not defined")
    return fn


class _DummyRewardContext:
    def __init__(self, episode_length: int):
        self.episode_length = int(episode_length)
        self._eg_noop_probe = True


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default
