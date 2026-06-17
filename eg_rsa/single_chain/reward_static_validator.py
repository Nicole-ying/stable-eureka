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
    """Extract the LAST (non-initialization) assignment to *name*.

    Reward codes often initialise a variable to 0.0 and then reassign it inside
    conditional blocks.  The first match is usually the dummy init, not the real
    reward logic, so we return the assignment whose RHS is most interesting
    (longest / most complex).
    """
    pattern = re.compile(r"^\s*" + re.escape(name) + r"\s*=\s*(.+)$", re.MULTILINE)
    matches = pattern.findall(code)
    if not matches:
        return ""
    # Pick the match with the longest non-trivial right-hand side — this is
    # almost always the semantic assignment rather than the zero-initialisation.
    best = max(matches, key=lambda m: (len(m.strip()), 1 if m.strip() in {"0", "0.0", "0.0;", "False", "None"} else 0))
    expr = best.strip()
    if expr.endswith(":"):
        expr = expr[:-1].strip()
    return expr[:500]


def _has_repeatable_action_bonus(code: str) -> bool:
    """Heuristic: any positive reward term that pays per-step without a one-shot gate.

    Looks for reward variables that get a positive value inside an unguarded
    per-step branch (no self._ flag, no terminal/event gate visible nearby).
    """
    # Find all reward variable names: reward_XXX that are assigned positive values
    reward_vars = set(re.findall(r"(reward_\w+)\s*=", code))
    # Also include known patterns from LLM-generated code
    bonus_patterns = [
        r"reward_main_engine\s*=\s*([0-9.]+)",
        r"reward_near_pad\s*=\s*([0-9.]+)",
        r"action_0_bonus\s*=\s*([0-9.]+)",
        r"controlled_descent_bonus\s*=\s*([0-9.]+)",
    ]
    has_one_shot = bool(re.search(r"self\._[A-Za-z]", code))

    for pat in bonus_patterns:
        m = re.search(pat, code)
        if m and float(m.group(1)) > 0 and not has_one_shot:
            return True

    # Also check generic reward_ variables assigned positive values without gate
    for var in reward_vars:
        # Find the context around each assignment
        for m in re.finditer(re.escape(var) + r"\s*=\s*([0-9.]+)", code):
            val = float(m.group(1))
            if val <= 0:
                continue
            # Check 3 lines before for a one-shot guard
            start = max(0, m.start() - 200)
            context = code[start:m.end()]
            if "self._" not in context and "terminated" not in context:
                return True

    return False
