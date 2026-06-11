from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from eg_rsa.reward.formula_validator import FormulaValidator


@dataclass
class BootstrapSchemaValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class BootstrapSchemaValidator:
    """Validate an LLM-generated initial reward schema before training."""

    SUPPORTED_COMPONENT_TYPES = {
        "formula_component",
        "conditional_formula_component",
        "action_penalty",
        "metric_value",
        "metric_delta",
        "metric_threshold_bonus",
        "metric_stagnation_penalty",
    }

    SUPPORTED_EVENT_TYPES = {
        "event_predicate",
        "event_bonus",
    }

    @classmethod
    def validate_schema(
        cls,
        schema: Dict[str, Any],
        primitive_interface: Dict[str, Any],
    ) -> BootstrapSchemaValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(schema, dict):
            return BootstrapSchemaValidationResult(False, ["schema must be a dict"], [])

        components = schema.get("components", [])
        event_rules = schema.get("event_rules", [])
        if not isinstance(components, list):
            errors.append("schema.components must be a list")
            components = []
        if not isinstance(event_rules, list):
            errors.append("schema.event_rules must be a list")
            event_rules = []

        allowed_vars = set(primitive_interface.get("allowed_formula_variables", []))
        allowed_funcs = set(primitive_interface.get("allowed_formula_functions", []))
        semantic_roles = set(primitive_interface.get("semantic_roles", []))
        forbidden_terms = [str(x).lower() for x in primitive_interface.get("forbidden_bootstrap_terms", [])]

        names = set()
        semantic_seen = set()

        def has_forbidden(text: Any) -> bool:
            s = str(text).lower()
            return any(term in s for term in forbidden_terms)

        for component in components:
            if not isinstance(component, dict):
                errors.append("Each component must be a dict")
                continue

            name = component.get("name")
            ctype = component.get("type")
            role = component.get("semantic_role")

            if not name:
                errors.append("Component missing name")
                continue
            if name in names:
                errors.append(f"Duplicate reward item name: {name}")
            names.add(name)

            if has_forbidden(name):
                errors.append(f"Component name contains forbidden term: {name}")

            if ctype not in cls.SUPPORTED_COMPONENT_TYPES:
                errors.append(f"Unsupported component type in bootstrap schema: {ctype}")

            if role:
                semantic_seen.add(role)
                if semantic_roles and role not in semantic_roles:
                    errors.append(f"Unsupported semantic_role for component {name}: {role}")
            else:
                warnings.append(f"Component {name} missing semantic_role")

            try:
                weight = float(component.get("weight", 1.0))
                if not math.isfinite(weight):
                    errors.append(f"Component {name} has non-finite weight")
            except Exception:
                errors.append(f"Component {name} has invalid weight")

            params = dict(component.get("params", {}) or {})
            formula = component.get("formula") or params.get("formula")
            condition = component.get("condition") or params.get("condition")

            if ctype in {"formula_component", "conditional_formula_component"}:
                if not formula:
                    errors.append(f"{ctype} {name} missing formula")
                else:
                    if has_forbidden(formula):
                        errors.append(f"Formula for {name} contains forbidden term")
                    result = FormulaValidator.validate_expression(str(formula), allowed_vars, allowed_funcs)
                    if not result.ok:
                        errors.extend([f"{name}.formula: {e}" for e in result.errors])

            if ctype == "conditional_formula_component":
                if not condition:
                    errors.append(f"conditional_formula_component {name} missing condition")
                else:
                    if has_forbidden(condition):
                        errors.append(f"Condition for {name} contains forbidden term")
                    result = FormulaValidator.validate_expression(str(condition), allowed_vars, allowed_funcs)
                    if not result.ok:
                        errors.extend([f"{name}.condition: {e}" for e in result.errors])

        for rule in event_rules:
            if not isinstance(rule, dict):
                errors.append("Each event_rule must be a dict")
                continue

            name = rule.get("name")
            rtype = rule.get("type")
            role = rule.get("semantic_role")

            if not name:
                errors.append("Event rule missing name")
                continue
            if name in names:
                errors.append(f"Duplicate reward item name: {name}")
            names.add(name)

            if has_forbidden(name):
                errors.append(f"Event rule name contains forbidden term: {name}")

            if rtype not in cls.SUPPORTED_EVENT_TYPES:
                errors.append(f"Unsupported event rule type in bootstrap schema: {rtype}")

            if role:
                semantic_seen.add(role)
                if semantic_roles and role not in semantic_roles:
                    errors.append(f"Unsupported semantic_role for event {name}: {role}")
            else:
                warnings.append(f"Event rule {name} missing semantic_role")

            condition = rule.get("condition", {})
            expr = None
            if isinstance(condition, str):
                expr = condition
            elif isinstance(condition, dict):
                expr = condition.get("expression") or condition.get("formula")
            else:
                errors.append(f"Event rule {name} condition must be string or dict")

            if rtype == "event_predicate":
                if not expr:
                    errors.append(f"event_predicate {name} missing condition expression")
                else:
                    if has_forbidden(expr):
                        errors.append(f"Condition for {name} contains forbidden term")
                    result = FormulaValidator.validate_expression(str(expr), allowed_vars, allowed_funcs)
                    if not result.ok:
                        errors.extend([f"{name}.condition: {e}" for e in result.errors])

        if "dense_guidance" not in semantic_seen:
            warnings.append("No dense_guidance component found")
        if not ({"terminal_success", "safety_constraint"} & semantic_seen):
            warnings.append("No terminal_success or safety_constraint reward item found")

        return BootstrapSchemaValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)


def write_validation_report(path: str, result: BootstrapSchemaValidationResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
