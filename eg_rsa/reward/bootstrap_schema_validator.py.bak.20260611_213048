from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eg_rsa.reward.formula_validator import FormulaValidator
from eg_rsa.reward.safe_formula_eval import safe_eval_formula


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
    """Validate LLM-generated V2 bootstrap output.

    This validator is intentionally formula-native and environment-agnostic.
    It checks safe syntax, schema structure, blueprint-schema consistency, and
    generic action-penalty sign safety. It does not contain task-specific banned
    component names or task-specific reward fixes.
    """

    SUPPORTED_COMPONENT_TYPES = {
        "formula_component",
        "conditional_formula_component",
        # legacy built-in; V2 LLM bootstrap should not attach custom formulas
        "action_penalty",
    }

    SUPPORTED_EVENT_TYPES = {
        "event_predicate",
    }

    @classmethod
    def validate_bootstrap_result(
        cls,
        bootstrap_result: Dict[str, Any],
        primitive_interface: Dict[str, Any],
    ) -> BootstrapSchemaValidationResult:
        if not isinstance(bootstrap_result, dict):
            return BootstrapSchemaValidationResult(False, ["bootstrap result must be a dict"], [])

        schema = bootstrap_result.get("initial_schema")
        blueprint = bootstrap_result.get("reward_blueprint")

        result = cls.validate_schema(schema, primitive_interface, reward_blueprint=blueprint)
        if not isinstance(blueprint, dict):
            result.errors.append("bootstrap result missing reward_blueprint dict")
            result.ok = False
        else:
            result.warnings.extend(cls._blueprint_warnings(blueprint, schema or {}))

        result.ok = len(result.errors) == 0
        return result

    @classmethod
    def validate_schema(
        cls,
        schema: Dict[str, Any],
        primitive_interface: Dict[str, Any],
        reward_blueprint: Optional[Dict[str, Any]] = None,
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

        names = set()
        semantic_seen = set()
        progress_like_count = 0
        terminal_count = 0

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

            if ctype not in cls.SUPPORTED_COMPONENT_TYPES:
                errors.append(
                    f"Unsupported component type in V2 bootstrap schema: {ctype}. "
                    f"Supported: {sorted(cls.SUPPORTED_COMPONENT_TYPES)}"
                )

            if role:
                semantic_seen.add(role)
                if semantic_roles and role not in semantic_roles:
                    errors.append(f"Unsupported semantic_role for component {name}: {role}")
            else:
                warnings.append(f"Component {name} missing semantic_role")

            if role in {"dense_guidance", "safety_constraint"}:
                progress_like_count += 1

            try:
                weight = float(component.get("weight", 1.0))
                if not math.isfinite(weight):
                    errors.append(f"Component {name} has non-finite weight")
            except Exception:
                errors.append(f"Component {name} has invalid weight")

            params = dict(component.get("params", {}) or {})
            formula = component.get("formula") or params.get("formula")
            condition = component.get("condition") or params.get("condition")

            if ctype in cls.SUPPORTED_COMPONENT_TYPES:
                if ctype == "action_penalty":
                    if formula:
                        warnings.append(
                            f"action_penalty {name} has custom formula; V2 normalizer "
                            "should convert it to formula_component with semantic_role=control_cost"
                        )
                    else:
                        warnings.append(
                            f"legacy action_penalty {name} has no custom formula; runtime uses built-in -sum(action^2)"
                        )
                elif not formula:
                    errors.append(f"{ctype} {name} missing formula")
                else:
                    validation = FormulaValidator.validate_expression(str(formula), allowed_vars, allowed_funcs)
                    if not validation.ok:
                        errors.extend([f"{name}.formula: {e}" for e in validation.errors])

            if ctype == "conditional_formula_component":
                if not condition:
                    errors.append(f"conditional_formula_component {name} missing condition")
                else:
                    validation = FormulaValidator.validate_expression(str(condition), allowed_vars, allowed_funcs)
                    if not validation.ok:
                        errors.extend([f"{name}.condition: {e}" for e in validation.errors])

            # action_penalty custom formulas are normalized to formula_component
            # before validation in normal V2 bootstrap flow. Do not hard-fail raw
            # bootstrap output here for a formula that runtime would ignore.

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

            if rtype not in cls.SUPPORTED_EVENT_TYPES:
                errors.append(
                    f"Unsupported event rule type in V2 bootstrap schema: {rtype}. "
                    f"Supported: {sorted(cls.SUPPORTED_EVENT_TYPES)}"
                )

            if role:
                semantic_seen.add(role)
                if semantic_roles and role not in semantic_roles:
                    errors.append(f"Unsupported semantic_role for event {name}: {role}")
            else:
                warnings.append(f"Event rule {name} missing semantic_role")

            if role == "terminal_success":
                terminal_count += 1

            condition = rule.get("condition", {})
            if isinstance(condition, str):
                expr = condition
            elif isinstance(condition, dict):
                expr = condition.get("expression") or condition.get("formula")
                if not expr:
                    expr = rule.get("expression") or rule.get("formula")
                    if expr:
                        warnings.append(
                            f"Event rule {name} used top-level expression/formula; "
                            "normalizer should move it into condition.expression"
                        )
            else:
                expr = rule.get("expression") or rule.get("formula")
                if expr:
                    warnings.append(
                        f"Event rule {name} missing condition dict but has top-level expression/formula"
                    )
                else:
                    expr = None
                    errors.append(f"Event rule {name} condition must be string or dict")

            if not expr:
                errors.append(f"event_predicate {name} missing condition expression")
            else:
                validation = FormulaValidator.validate_expression(str(expr), allowed_vars, allowed_funcs)
                if not validation.ok:
                    errors.extend([f"{name}.condition: {e}" for e in validation.errors])

            if isinstance(condition, dict) and "duration_steps" in condition:
                try:
                    if int(condition["duration_steps"]) <= 0:
                        errors.append(f"Event rule {name} has invalid duration_steps")
                except Exception:
                    errors.append(f"Event rule {name} has non-integer duration_steps")

        if "dense_guidance" not in semantic_seen:
            warnings.append("No dense_guidance component found")

        if not ({"terminal_success", "safety_constraint"} & semantic_seen):
            warnings.append("No terminal_success or safety_constraint reward item found")

        if terminal_count <= 0:
            warnings.append("No terminal_success event_predicate found")

        if progress_like_count <= 0:
            warnings.append("No progress/control process component found")

        if reward_blueprint is not None:
            warnings.extend(cls._blueprint_warnings(reward_blueprint, schema))

        return BootstrapSchemaValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)

    @staticmethod
    def _check_action_penalty_sign_safety(
        name: str,
        formula: str,
        weight: Any,
        allowed_vars: set[str],
    ) -> str:
        try:
            w = float(weight)
        except Exception:
            return ""

        base = {var: 0.0 for var in allowed_vars}
        samples = [
            {"main_engine": 0.0, "side_engine": 0.0},
            {"main_engine": 1.0, "side_engine": 0.0},
            {"main_engine": 0.0, "side_engine": 1.0},
            {"main_engine": 0.0, "side_engine": -1.0},
            {"main_engine": 1.0, "side_engine": 1.0},
            {"main_engine": 1.0, "side_engine": -1.0},
        ]

        for sample in samples:
            variables = dict(base)
            variables.update(sample)
            try:
                raw = float(safe_eval_formula(formula, variables=variables))
            except Exception:
                continue
            value = w * raw
            if value > 1e-9:
                return (
                    f"action_penalty {name} can produce positive reward under sampled action "
                    f"{sample}: weight * formula = {value:.6g}. "
                    "Action penalties must not reward action direction."
                )
        return ""

    @staticmethod
    def _blueprint_warnings(blueprint: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        if not isinstance(blueprint, dict):
            warnings.append("reward_blueprint missing or not a dict")
            return warnings

        roles = blueprint.get("primitive_variable_roles", {})
        if not isinstance(roles, dict):
            warnings.append("reward_blueprint.primitive_variable_roles missing or not a dict")

        phases = blueprint.get("phase_structure", [])
        if not isinstance(phases, list) or not phases:
            warnings.append("reward_blueprint.phase_structure missing or empty")

        component_blueprint = blueprint.get("component_blueprint", [])
        if not isinstance(component_blueprint, list) or not component_blueprint:
            warnings.append("reward_blueprint.component_blueprint missing or empty")

        schema_names = {
            item.get("name")
            for item in list(schema.get("components", []) or []) + list(schema.get("event_rules", []) or [])
            if isinstance(item, dict)
        }

        blueprint_names = {
            item.get("name")
            for item in component_blueprint
            if isinstance(item, dict) and item.get("name")
        }

        missing = sorted(name for name in blueprint_names if name not in schema_names)
        if missing:
            warnings.append(f"Blueprint names not present in schema: {missing}")

        return warnings


def write_validation_report(path: str, result: BootstrapSchemaValidationResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
