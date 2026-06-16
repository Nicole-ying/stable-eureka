from __future__ import annotations

import re
from typing import Any, Dict, List


def validate_reward_static(
    reward_code: str,
    reward_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Static checks for common reward-payment failure modes.

    This validator is intentionally conservative: it only blocks patterns that are
    strongly associated with reward hacking in the 20x1M LunarLander run, while
    reporting softer concerns as warnings.
    """
    code = reward_code or ""
    errors: List[str] = []
    warnings: List[str] = []

    success_expr = _extract_assignment_expression(code, "success")
    objective_expr = _extract_assignment_expression(code, "objective_bonus")
    reward_expr = _extract_assignment_expression(code, "reward")

    has_one_shot_state = bool(re.search(r"self\._[A-Za-z0-9_]*(success|objective|terminal)[A-Za-z0-9_]*", code))
    objective_uses_success = "success" in objective_expr or "success_flag" in objective_expr
    objective_direct_success = bool(re.search(r"objective_bonus\s*=\s*[0-9.]+\s+if\s+success\s+else\s+0", code))
    objective_direct_success_flag = bool(re.search(r"objective_bonus\s*=\s*[0-9.]+\s+if\s+success_flag\s+else\s+0", code))

    if objective_direct_success and not has_one_shot_state:
        errors.append(
            "objective_bonus appears to pay a large reward every step that success is true; use terminal alignment or a one-shot guard."
        )

    if objective_direct_success_flag and not has_one_shot_state:
        errors.append(
            "objective_bonus depends on success_flag without a visible one-shot guard; this can repeat many times per episode."
        )

    if objective_uses_success and success_expr and "terminated" not in success_expr and not has_one_shot_state:
        errors.append(
            "success used by objective_bonus is a non-terminal state predicate and is not visibly one-shot guarded."
        )
    elif objective_uses_success and success_expr and "terminated" not in success_expr:
        warnings.append(
            "success used by objective_bonus is not terminal-conditioned; the one-shot guard may prevent farming but terminal alignment should still be checked."
        )

    if _has_repeatable_action_bonus(code):
        warnings.append(
            "Raw action/control bonus appears repeatable; ensure it is phase-conditioned, progress-conditioned, or very small."
        )

    if "individual_reward" not in code:
        warnings.append("reward_code does not expose individual_reward diagnostics; audits will be less informative.")

    valid = not errors
    return {
        "file_type": "reward_static_validation",
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "success_expression": success_expr,
            "objective_bonus_expression": objective_expr,
            "reward_expression": reward_expr,
            "has_one_shot_state": has_one_shot_state,
            "objective_uses_success": objective_uses_success,
            "objective_direct_success": objective_direct_success,
            "objective_direct_success_flag": objective_direct_success_flag,
        },
    }


def _extract_assignment_expression(code: str, name: str) -> str:
    pattern = re.compile(r"^\s*" + re.escape(name) + r"\s*=\s*(.+)$", re.MULTILINE)
    match = pattern.search(code)
    if not match:
        return ""
    expr = match.group(1).strip()
    if expr.endswith(":"):
        expr = expr[:-1].strip()
    return expr[:500]


def _has_repeatable_action_bonus(code: str) -> bool:
    suspicious_names = [
        "action_0_bonus",
        "action_bonus",
        "main_engine_bonus",
        "main_engine_usage",
        "controlled_descent_bonus",
    ]
    for name in suspicious_names:
        expr = _extract_assignment_expression(code, name)
        if not expr:
            continue
        if "if" in expr and "self._" not in code:
            return True
    return False
