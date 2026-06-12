from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eg_rsa.reward.formula_ast import validate_formula_ast


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
    """Validate AST-first V2 bootstrap output."""

    SUPPORTED_COMPONENT_TYPES = {
        "formula_component",
        "conditional_formula_component",
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
            params = dict(component.get("params", {}) or {})

            if not name:
                errors.append("Component missing name")
                continue

            if name in names:
                errors.append(f"Duplicate reward item name: {name}")
            names.add(name)

            if ctype not in cls.SUPPORTED_COMPONENT_TYPES:
                errors.append(
                    f"Unsupported AST component type: {ctype}. "
                    f"Supported: {sorted(cls.SUPPORTED_COMPONENT_TYPES)}"
                )

            if role:
                semantic_seen.add(role)
                if semantic_roles and role not in semantic_roles:
                    errors.append(f"Unsupported semantic_role for component {name}: {role}")
            else:
                warnings.append(f"Component {name} missing semantic_role")

            if role in {"dense_guidance", "safety_constraint", "stability_quality"}:
                progress_like_count += 1

            try:
                weight = float(component.get("weight", 1.0))
                if not math.isfinite(weight):
                    errors.append(f"Component {name} has non-finite weight")
            except Exception:
                errors.append(f"Component {name} has invalid weight")

            clip = component.get("clip")
            if clip is not None and not cls._is_valid_clip(clip):
                errors.append(f"Component {name} has invalid clip range: {clip}")

            formula_ast = component.get("formula_ast") or params.get("formula_ast")
            condition_ast = component.get("condition_ast") or params.get("condition_ast")

            if ctype in cls.SUPPORTED_COMPONENT_TYPES:
                if formula_ast is None:
                    errors.append(f"{ctype} {name} missing formula_ast")
                else:
                    validation = validate_formula_ast(formula_ast, allowed_vars)
                    if not validation.ok:
                        errors.extend([f"{name}.formula_ast: {e}" for e in validation.errors])

            if ctype == "conditional_formula_component":
                if condition_ast is None:
                    errors.append(f"conditional_formula_component {name} missing condition_ast")
                else:
                    validation = validate_formula_ast(condition_ast, allowed_vars)
                    if not validation.ok:
                        errors.extend([f"{name}.condition_ast: {e}" for e in validation.errors])

            if "formula" in component or "formula" in params:
                errors.append(f"{name}: string formula is forbidden in AST-first schema; use formula_ast")

            if "condition" in component or "condition" in params:
                errors.append(f"{name}: string condition is forbidden in AST-first schema; use condition_ast")

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
                    f"Unsupported AST event rule type: {rtype}. "
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
            if not isinstance(condition, dict):
                errors.append(f"Event rule {name} condition must be dict")
                continue

            expr_ast = rule.get("condition_ast") or rule.get("expr_ast") or condition.get("expr_ast") or condition.get("condition_ast")
            if expr_ast is None:
                errors.append(f"event_predicate {name} missing condition.expr_ast")
            else:
                validation = validate_formula_ast(expr_ast, allowed_vars)
                if not validation.ok:
                    errors.extend([f"{name}.condition_ast: {e}" for e in validation.errors])

            if "expression" in condition or "formula" in condition or "expression" in rule or "formula" in rule:
                errors.append(f"{name}: string event expression is forbidden in AST-first schema; use condition.expr_ast")

            if "duration_steps" in condition:
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
    def _is_valid_clip(clip: Any) -> bool:
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:
            return False
        try:
            low = float(clip[0])
            high = float(clip[1])
        except Exception:
            return False
        return math.isfinite(low) and math.isfinite(high) and low <= high

    @staticmethod
    def _blueprint_warnings(blueprint: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        if not isinstance(blueprint, dict):
            warnings.append("reward_blueprint missing or not a dict")
            return warnings

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
