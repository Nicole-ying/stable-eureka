from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Tuple


SUCCESS_NAMES = {"success", "success_flag", "safe_landing", "landed", "is_success"}
OBJECTIVE_NAMES = {"objective_bonus", "reward_success", "success_bonus", "terminal_reward"}
INIT_VALUES = {"0", "0.0", "False", "None"}


def validate_reward_static(
    reward_code: str,
    reward_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Static checks for common reward-payment and evidence-contract failures.

    This validator is not the expert search brain.  It enforces structural I/O
    contracts and surfaces reward-design risks.  Search decisions still belong to
    the LLM expert strategist.
    """
    code = reward_code or ""
    reward_schema = reward_schema or {}
    errors: List[str] = []
    warnings: List[str] = []

    ast_report = _assignment_context_report(code)
    success_assignment = _select_assignment(ast_report, SUCCESS_NAMES)
    objective_assignment = _select_assignment(ast_report, OBJECTIVE_NAMES)
    reward_assignment = _select_assignment(ast_report, {"reward"})

    success_expr = success_assignment.get("rhs", "")
    objective_expr = objective_assignment.get("rhs", "")
    reward_expr = reward_assignment.get("rhs", "")
    objective_context = " and ".join(objective_assignment.get("guards", []) or [])
    success_context = " and ".join(success_assignment.get("guards", []) or [])
    objective_full_context = " ".join([objective_context, objective_expr])
    success_full_context = " ".join([success_context, success_expr])

    has_one_shot_state = bool(re.search(r"self\._[A-Za-z0-9_]*(success|objective|terminal)[A-Za-z0-9_]*", code))
    objective_mentions_success = any(name in objective_full_context for name in SUCCESS_NAMES)
    objective_terminal_guarded = "terminated" in objective_full_context or "done" in objective_full_context
    success_terminal_guarded = "terminated" in success_full_context or "done" in success_full_context
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

    if objective_mentions_success and not objective_terminal_guarded and not success_terminal_guarded and not has_one_shot_state:
        errors.append(
            "objective/success reward appears tied to a non-terminal success predicate without a visible one-shot guard."
        )
    elif objective_mentions_success and not objective_terminal_guarded and not success_terminal_guarded:
        warnings.append(
            "objective/success reward is not visibly terminal-conditioned; one-shot state may prevent farming but terminal alignment should be checked."
        )

    if _has_repeatable_action_bonus(code):
        warnings.append(
            "Raw action/control bonus appears repeatable; ensure it is phase-conditioned, progress-conditioned, or very small."
        )

    if "individual_reward" not in code:
        warnings.append("reward_code does not expose individual_reward diagnostics; audits will be less informative.")

    schema_consistency = _schema_code_consistency(reward_schema, code)
    errors.extend(schema_consistency.get("errors", []))
    warnings.extend(schema_consistency.get("warnings", []))

    valid = not errors
    return {
        "file_type": "reward_static_validation",
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "success_expression": success_expr,
            "success_assignment_guards": success_assignment.get("guards", []),
            "objective_bonus_expression": objective_expr,
            "objective_bonus_assignment_guards": objective_assignment.get("guards", []),
            "reward_expression": reward_expr,
            "has_one_shot_state": has_one_shot_state,
            "objective_mentions_success": objective_mentions_success,
            "objective_terminal_guarded": objective_terminal_guarded,
            "success_terminal_guarded": success_terminal_guarded,
            "objective_direct_success": objective_direct_success,
            "objective_direct_success_flag": objective_direct_success_flag,
            "schema_code_consistency": schema_consistency,
            "assignment_contexts": ast_report.get("assignments", []),
        },
    }


def _assignment_context_report(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"valid_ast": False, "parse_error": str(exc), "assignments": []}

    assignments: List[Dict[str, Any]] = []

    def visit(node: ast.AST, guards: List[str]) -> None:
        if isinstance(node, ast.If):
            test = _unparse(node.test)
            for child in node.body:
                visit(child, guards + [test])
            for child in node.orelse:
                visit(child, guards + ["not (" + test + ")"])
            return
        if isinstance(node, ast.Assign):
            rhs = _unparse(node.value)
            for target in node.targets:
                name = _target_name(target)
                if name:
                    assignments.append({"name": name, "rhs": rhs, "guards": guards, "lineno": getattr(node, "lineno", None)})
        for child in ast.iter_child_nodes(node):
            visit(child, guards)

    visit(tree, [])
    return {"valid_ast": True, "assignments": assignments}


def _select_assignment(report: Dict[str, Any], names: set[str]) -> Dict[str, Any]:
    candidates = [a for a in report.get("assignments", []) if str(a.get("name")) in names]
    if not candidates:
        return {}

    def score(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
        rhs = str(item.get("rhs", "")).strip()
        guards = item.get("guards", []) or []
        non_init = 0 if rhs in INIT_VALUES else 1
        terminal = 1 if "terminated" in " ".join(guards + [rhs]) else 0
        success = 1 if any(tok in " ".join(guards + [rhs]) for tok in SUCCESS_NAMES) else 0
        return (non_init, terminal, success, len(rhs) + 30 * len(guards))

    return max(candidates, key=score)


def _has_repeatable_action_bonus(code: str) -> bool:
    """Heuristic: positive per-step control bonuses without visible event/phase gate."""
    reward_vars = set(re.findall(r"(reward_\w+)\s*=", code))
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

    for var in reward_vars:
        for m in re.finditer(re.escape(var) + r"\s*=\s*([0-9.]+)", code):
            val = float(m.group(1))
            if val <= 0:
                continue
            start = max(0, m.start() - 220)
            context = code[start:m.end()]
            if "self._" not in context and "terminated" not in context and "progress" not in context:
                return True
    return False


def _schema_code_consistency(reward_schema: Dict[str, Any], code: str) -> Dict[str, Any]:
    component_names = _schema_component_names(reward_schema)
    diagnostic_keys = _extract_individual_reward_keys(code)
    component_set = set(component_names)
    key_set = set(diagnostic_keys)
    missing_from_code = sorted(component_set - key_set)
    extra_in_code = sorted(key_set - component_set)
    errors: List[str] = []
    warnings: List[str] = []
    if component_names and diagnostic_keys and missing_from_code:
        errors.append(
            "reward_schema component ids are not exposed in individual_reward diagnostics: " + ", ".join(missing_from_code[:12])
        )
    if component_names and diagnostic_keys and extra_in_code:
        errors.append(
            "individual_reward exposes keys not declared by reward_schema component ids: " + ", ".join(extra_in_code[:12])
        )
    if component_names and not diagnostic_keys:
        warnings.append("reward_schema declares components but individual_reward keys could not be extracted statically.")
    return {
        "schema_component_names": component_names,
        "individual_reward_keys": diagnostic_keys,
        "missing_schema_components_in_code": missing_from_code,
        "extra_code_components_not_in_schema": extra_in_code,
        "errors": errors,
        "warnings": warnings,
    }


def _schema_component_names(reward_schema: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for container in ["component_catalog", "components", "active_reward_terms", "diagnostic_terms"]:
        value = reward_schema.get(container)
        if not isinstance(value, list):
            continue
        for item in value:
            name = None
            if isinstance(item, dict):
                name = item.get("id") or item.get("name")
            elif isinstance(item, str):
                name = item
            if name and name not in names:
                names.append(str(name))
    return names


def _extract_individual_reward_keys(code: str) -> List[str]:
    keys: List[str] = []
    for match in re.finditer(r"individual_reward\[['\"]([^'\"]+)['\"]\]", code):
        key = match.group(1)
        if key not in keys:
            keys.append(key)
    dict_match = re.search(r"individual_reward\s*=\s*\{(?P<body>.*?)\}\s*", code, re.S)
    if dict_match:
        body = dict_match.group("body")
        for match in re.finditer(r"['\"]([^'\"]+)['\"]\s*:", body):
            key = match.group(1)
            if key not in keys:
                keys.append(key)
    return keys


def _target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Subscript):
        root = _target_name(target.value)
        sub = _unparse(target.slice)
        return f"{root}[{sub}]" if root else ""
    if isinstance(target, ast.Attribute):
        return _unparse(target)
    return ""


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""
