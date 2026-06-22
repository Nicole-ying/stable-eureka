from __future__ import annotations

import ast
import difflib
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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

SAFE_BUILTINS = {
    "abs": abs,
    "float": float,
    "int": int,
    "min": min,
    "max": max,
    "bool": bool,
    "hasattr": hasattr,
}
ALLOWED_CALL_ROOTS = set(SAFE_BUILTINS.keys())
ALLOWED_MODULE_CALLS = {"math", "np", "numpy"}
DISALLOWED_NODES = (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Raise, ast.Lambda, ast.Global, ast.Nonlocal)


def detect_semantic_noop_edit(
    old_code: str | None,
    new_code: str | None,
    probes: List[ProbeCase] | None = None,
    reward_abs_tol: float = 1e-6,
    component_abs_tol: float = 1e-6,
) -> Dict[str, Any]:
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

    semantic_noop = bool(
        ast_equal
        or (source_similarity >= 0.985 and outputs_equivalent)
        or (source_similarity >= 0.94 and outputs_equivalent and probe_report.get("component_keys_equivalent"))
    )
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
    namespace: Dict[str, Any] = {"math": math, "np": np, "numpy": np, "__builtins__": SAFE_BUILTINS}
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


# --- Pattern-based edit repetition detection ---

# Known anti-pattern fingerprints extracted from failed memory entries.
# Each pattern has a name, code-level signatures to match against new code,
# and a severity level.
FAILED_PATTERN_SIGNATURES: List[Dict[str, Any]] = [
    {
        "pattern_id": "boundary_or_position_penalty_without_centering",
        "description": "Penalizing horizontal position (state[0]) without providing a positive centering signal",
        "severity": "HARD",
        "code_signatures": [
            r"abs\(.*\b(state\[0\]|x_pos|x\b).*\).*\s*[>\-]",
            r"out.of.bounds.*-",
        ],
        "structural_check": "penalty_on_x_without_centering",
    },
    {
        "pattern_id": "reduced_crash_penalty_during_timeout",
        "description": "Reducing crash penalty magnitude when timeout is the dominant failure mode",
        "severity": "STRONG",
        "code_signatures": [
            r"crash_penalty\s*=\s*-(?:50|25|10|5|0)\b(?!\d)",
        ],
    },
    {
        "pattern_id": "removed_all_progress_signals",
        "description": "Removing all dense progress signals and replacing with a single one-shot bonus",
        "severity": "STRONG",
        "code_signatures": [],
        "structural_check": "count_progress_signals",
    },
]

# Component categories used for structural analysis
PROGRESS_COMPONENT_KEYWORDS = [
    "stability", "descent", "progress", "approach", "centering",
    "distance", "velocity", "height", "alignment", "orientation",
]


def _build_dynamic_patterns(
    memory_context: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build pattern signatures dynamically from memory context.

    These are informational — they document known failure modes but don't have
    code-level signatures for automated detection. HARD blocks come from
    the hand-crafted FAILED_PATTERN_SIGNATURES.
    """
    if not memory_context:
        return []

    dynamic: List[Dict[str, Any]] = []
    negative_edges = memory_context.get("negative_edges") or []
    if not isinstance(negative_edges, list):
        return []

    for edge in negative_edges:
        tier = str(edge.get("tier", "")).upper()
        if tier not in ("HARD", "STRONG"):
            continue
        avoid = str(edge.get("avoid_pattern", ""))
        if not avoid or len(avoid) < 10:
            continue

        # Dynamic patterns serve as documentation for the pattern_repeat_report.
        # The `from_memory` flag tells consumers this came from live experiment data.
        dynamic.append({
            "pattern_id": f"memory_edge_{edge.get('from','?')}_to_{edge.get('to','?')}",
            "description": avoid[:200],
            "severity": "STRONG",
            "code_signatures": [],
            "structural_check": "",
            "from_memory": True,
        })

    return dynamic


def detect_pattern_repeat(
    new_code: str,
    memory_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect if the new code repeats a known failed edit pattern.

    This is an additional layer beyond AST/probe-based noop detection.
    It checks structural edit patterns that are known to fail, even if
    the code text differs significantly.

    Args:
        new_code: The newly generated reward code.
        memory_context: Optional memory context with negative_edges and hard_constraints.

    Returns:
        Dict with 'pattern_repeat_detected', 'matched_patterns', and 'should_block'.
    """
    if not new_code or not new_code.strip():
        return _clean_pattern_report()

    code_lower = new_code.lower()
    matched: List[Dict[str, Any]] = []

    # Build dynamic patterns from memory context's negative edges
    dynamic_patterns = _build_dynamic_patterns(memory_context)
    all_patterns = list(FAILED_PATTERN_SIGNATURES) + dynamic_patterns

    for pattern in all_patterns:
        sig_matches = []
        for sig in pattern.get("code_signatures", []):
            if re.search(sig, new_code, re.IGNORECASE):
                sig_matches.append(sig)
        structural_match = False
        structural_detail = ""
        structural_check = pattern.get("structural_check", "")
        if structural_check == "count_progress_signals":
            # Check if the code has zero positive progress-like components
            progress_count = _count_progress_signals(new_code)
            if progress_count == 0:
                structural_match = True
                structural_detail = f"No progress-like signals detected in code (expected at least 1)"
        elif structural_check == "penalty_on_x_without_centering":
            # Check if state[0] is used in a penalty context AND no centering signal exists
            has_x_penalty = bool(re.search(r'(?:abs\(|-\s*\(?\s*\d).*\b(?:state\[0\]|x_pos|x\b)', new_code, re.IGNORECASE))
            has_centering = bool(re.search(r'centering|approach.*pad|stay.*center|near.*pad|abs.*x.*<\s*\d.*reward|reward.*abs.*x', new_code, re.IGNORECASE))
            if has_x_penalty and not has_centering:
                structural_match = True
                structural_detail = "Penalizes horizontal position but has no positive centering/proximity signal"
        elif structural_check.startswith("keyword_check:"):
            # Dynamic keyword check from memory context
            # Higher threshold for HARD patterns, lower for STRONG
            kw_list = structural_check[len("keyword_check:"):].split("|")
            kw_list = [kw for kw in kw_list if kw]  # filter empty
            if not kw_list:
                continue
            match_count = sum(1 for kw in kw_list if kw and kw in code_lower)
            # Require at least 60% keyword match + minimum of 2 matches
            min_matches = max(2, int(len(kw_list) * 0.6))
            if match_count >= min_matches:
                structural_match = True
                structural_detail = f"Code contains {match_count}/{len(kw_list)} keywords from forbidden pattern: {', '.join(kw_list[:5])}"
        # Dynamic patterns from memory are always included as informational context.
        # They document known failure modes but rely on human/code review for enforcement.
        is_dynamic_memory_pattern = pattern.get("from_memory") or pattern["pattern_id"].startswith("memory_")
        has_no_checks = not sig_matches and not structural_match

        if is_dynamic_memory_pattern and has_no_checks:
            matched.append({
                "pattern_id": pattern["pattern_id"],
                "description": pattern["description"],
                "severity": "STRONG",
                "signature_matches": [],
                "structural_match": False,
                "structural_detail": "Pattern from experiment memory — review before training",
                "from_memory": True,
            })
        elif sig_matches or structural_match:
            matched.append({
                "pattern_id": pattern["pattern_id"],
                "description": pattern["description"],
                "severity": pattern["severity"],
                "signature_matches": sig_matches,
                "structural_match": structural_match,
                "structural_detail": structural_detail,
            })

    # Check against memory context hard constraints
    memory_matches = []
    if memory_context:
        hard_constraints = memory_context.get("hard_constraints_for_next_revision", []) or []
        for constraint in hard_constraints:
            constraint_lower = str(constraint).lower()
            # Check if the new code might violate known constraints
            if "do not" in constraint_lower or "avoid" in constraint_lower or "forbidden" in constraint_lower:
                # Extract key phrases and check against code
                keywords = [w for w in constraint_lower.split() if len(w) >= 5]
                matching_keywords = [kw for kw in keywords if kw in code_lower]
                if len(matching_keywords) >= 3:
                    memory_matches.append({
                        "constraint": str(constraint)[:300],
                        "matching_keywords": matching_keywords,
                    })

    has_hard_violation = any(m["severity"] == "HARD" for m in matched)
    has_pattern_repeat = len(matched) > 0

    return {
        "file_type": "pattern_repeat_report",
        "pattern_repeat_detected": has_pattern_repeat,
        "matched_patterns": matched,
        "memory_constraint_matches": memory_matches,
        "should_block": has_hard_violation,
        "hard_constraints": [
            f"Pattern repeat detected: {m['description']}"
            for m in matched if m["severity"] == "HARD"
        ],
        "warnings": [
            f"Possible pattern repeat: {m['description']}"
            for m in matched if m["severity"] != "HARD"
        ],
    }


def _count_progress_signals(code: str) -> int:
    """Count likely progress signal components in reward code."""
    count = 0
    for kw in PROGRESS_COMPONENT_KEYWORDS:
        pattern = rf'{kw}\w*\s*[=+]'
        if re.search(pattern, code, re.IGNORECASE):
            count += 1
    return max(0, count)


def _clean_pattern_report() -> Dict[str, Any]:
    return {
        "file_type": "pattern_repeat_report",
        "pattern_repeat_detected": False,
        "matched_patterns": [],
        "memory_constraint_matches": [],
        "should_block": False,
        "hard_constraints": [],
        "warnings": [],
    }
