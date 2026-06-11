from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eg_rsa.reward.formula_ast import validate_formula_ast
from eg_rsa.reward.operators import RewardEditOperatorApplier
from eg_rsa.reward.schema import RewardSchema


@dataclass
class EditPlanValidationResult:
    valid_edits: List[Dict[str, Any]] = field(default_factory=list)
    rejected_edits: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.valid_edits) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "valid_edits": self.valid_edits,
            "rejected_edits": self.rejected_edits,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class EditPlanValidator:
    """Validate AST-first edit plans before schema mutation."""

    DEFAULT_ALLOWED_VARS = {
        "x", "y", "vx", "vy", "angle", "angular_velocity",
        "left_contact", "right_contact", "main_engine", "side_engine",
        "contact", "both_contact",
    }

    @classmethod
    def validate(
        cls,
        schema: RewardSchema,
        edit_plan: List[Dict[str, Any]],
        structural_context: Optional[Dict[str, Any]] = None,
    ) -> EditPlanValidationResult:
        structural_context = structural_context or {}
        result = EditPlanValidationResult()

        if not isinstance(edit_plan, list):
            result.errors.append("edit_plan must be a list")
            return result

        for idx, edit in enumerate(edit_plan):
            normalized, notes = cls._normalize_edit(edit, structural_context)
            result.warnings.extend(notes)
            ok, error = cls._validate_one(schema, normalized, structural_context)
            if ok:
                result.valid_edits.append(normalized)
            else:
                rejected = dict(normalized) if isinstance(normalized, dict) else {"raw_edit": normalized}
                rejected["index"] = idx
                result.rejected_edits.append(rejected)
                result.errors.append(error)

        return result

    @classmethod
    def safe_fallback(cls, schema: RewardSchema, diagnostics: Dict[str, Any]) -> List[Dict[str, Any]]:
        target = diagnostics.get("dominant_component")
        modes = set(diagnostics.get("failure_modes", []))

        if target and schema.get_component(target):
            if "single_component_dominance" in modes:
                return [{"operator": "decrease_weight", "target": target, "factor": 0.5}]
            return [{"operator": "clip_component", "target": target, "clip": [-1.0, 1.0]}]

        for component in schema.components:
            if component.enabled:
                return [{"operator": "clip_component", "target": component.name, "clip": [-1.0, 1.0]}]

        return []

    @classmethod
    def _normalize_edit(cls, edit: Any, structural_context: Dict[str, Any]) -> Tuple[Any, List[str]]:
        if not isinstance(edit, dict):
            return edit, []

        op = edit.get("operator") or edit.get("op")

        if op in {"add_formula_component", "add_conditional_formula_component", "add_action_penalty"}:
            normalized = dict(edit)
            component = dict(normalized.get("component", {}) or {})
            params = dict(component.get("params", {}) or {})

            if "formula_ast" in component and "formula_ast" not in params:
                params["formula_ast"] = component["formula_ast"]
            if "condition_ast" in component and "condition_ast" not in params:
                params["condition_ast"] = component["condition_ast"]

            component["params"] = params
            component.setdefault("inputs", [])
            component.setdefault("enabled", True)

            if op == "add_formula_component":
                component["type"] = "formula_component"
            elif op == "add_conditional_formula_component":
                component["type"] = "conditional_formula_component"
            elif op == "add_action_penalty":
                component["type"] = "formula_component"
                component.setdefault("semantic_role", "control_cost")
                component.setdefault("reward_timing", "dense")
                component.setdefault("behavior_channel", "control")

            normalized["component"] = component
            normalized["operator"] = "add_component"
            return normalized, []

        if op == "add_event_predicate":
            normalized = dict(edit)
            rule = dict(normalized.get("event_rule", {}) or {})
            rule["type"] = "event_predicate"
            rule.setdefault("enabled", True)
            rule.setdefault("one_time", True)
            condition = rule.get("condition", {})
            condition = dict(condition or {}) if isinstance(condition, dict) else {}
            if "condition_ast" in rule and "expr_ast" not in condition:
                condition["expr_ast"] = rule["condition_ast"]
            condition.setdefault("duration_steps", int(rule.get("duration_steps", condition.get("duration_steps", 1)) or 1))
            rule["condition"] = condition
            normalized["event_rule"] = rule
            normalized["operator"] = "add_event_rule"
            return normalized, []

        if op == "replace_formula":
            normalized = dict(edit)
            normalized["operator"] = op
            return normalized, []

        if op == "replace_condition":
            normalized = dict(edit)
            normalized["operator"] = op
            if "condition_ast" not in normalized and isinstance(normalized.get("condition"), dict):
                ast_node = normalized["condition"].get("expr_ast") or normalized["condition"].get("condition_ast")
                if ast_node is not None:
                    normalized["condition_ast"] = ast_node
            return normalized, []

        return edit, []

    @classmethod
    def _validate_one(cls, schema: RewardSchema, edit: Any, structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(edit, dict):
            return False, "edit must be a dict"

        op = edit.get("operator") or edit.get("op")
        if op not in RewardEditOperatorApplier.ALLOWED_OPERATORS:
            return False, f"unsupported operator: {op}"

        if op in {"increase_weight", "decrease_weight", "clip_component", "disable_component", "convert_to_one_time_event", "add_duration_condition", "replace_formula", "replace_condition"}:
            target = edit.get("target")
            if not target:
                return False, f"{op} requires target"
            if schema.get_component(target) is None and schema.get_event_rule(target) is None:
                return False, f"target not found: {target}"

        if op == "increase_weight" and float(edit.get("factor", 0.0)) <= 1.0:
            return False, "increase_weight requires factor > 1"

        if op == "decrease_weight":
            factor = float(edit.get("factor", 0.0))
            if not 0.0 < factor < 1.0:
                return False, "decrease_weight requires 0 < factor < 1"

        if op == "clip_component":
            clip = edit.get("clip")
            if not isinstance(clip, list) or len(clip) != 2 or float(clip[0]) > float(clip[1]):
                return False, "clip_component requires clip=[min,max]"
            if schema.get_component(edit.get("target")) is None:
                return False, "clip_component target must be a component"

        if op == "replace_formula":
            return cls._validate_replace_formula(schema, edit, structural_context)

        if op == "replace_condition":
            return cls._validate_replace_condition(schema, edit, structural_context)

        if op == "add_component":
            return cls._validate_add_component(schema, edit, structural_context)

        if op == "add_event_rule":
            return cls._validate_add_event_rule(schema, edit, structural_context)

        if op == "add_duration_condition":
            if schema.get_event_rule(edit.get("target")) is None:
                return False, "add_duration_condition target must be an event rule"
            if int(edit.get("duration_steps", 0)) <= 0:
                return False, "duration_steps must be positive"

        if op == "convert_to_one_time_event":
            target = edit.get("target")
            component = schema.get_component(target)
            rule = schema.get_event_rule(target)
            if component is None and rule is None:
                return False, f"target not found: {target}"
            if component is not None and component.type != "event_bonus":
                return False, "convert_to_one_time_event requires an event_bonus component or event rule"

        return True, ""

    @classmethod
    def _validate_replace_formula(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        component = schema.get_component(edit.get("target"))
        if component is None:
            return False, "replace_formula target must be a component"
        if component.type not in RewardEditOperatorApplier.FORMULA_COMPONENT_TYPES:
            return False, f"replace_formula target must be formula-native component, got {component.type}"
        ast_node = edit.get("formula_ast")
        if ast_node is None:
            return False, "replace_formula requires formula_ast"
        err = cls._validate_ast_text(ast_node, structural_context)
        if err:
            return False, f"replace_formula formula_ast invalid: {err}"
        if "formula" in edit:
            return False, "replace_formula string formula is forbidden; use formula_ast"
        return True, ""

    @classmethod
    def _validate_replace_condition(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        target = edit.get("target")
        component = schema.get_component(target)
        rule = schema.get_event_rule(target)
        ast_node = edit.get("condition_ast")

        if ast_node is None:
            return False, "replace_condition requires condition_ast"

        if component is not None:
            if component.type != "conditional_formula_component":
                return False, "replace_condition component target must be conditional_formula_component"
            err = cls._validate_ast_text(ast_node, structural_context)
            if err:
                return False, f"replace_condition condition_ast invalid: {err}"
            return True, ""

        if rule is not None:
            if rule.type != "event_predicate":
                return False, "replace_condition event target must be event_predicate"
            err = cls._validate_ast_text(ast_node, structural_context)
            if err:
                return False, f"replace_condition condition_ast invalid: {err}"
            return True, ""

        return False, "replace_condition target not found"

    @classmethod
    def _validate_add_component(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        component = edit.get("component")
        if not isinstance(component, dict):
            return False, "add_component requires component dict"

        name = component.get("name")
        if not name:
            return False, "new component requires name"

        if schema.get_component(name) or schema.get_event_rule(name):
            return False, f"reward item already exists: {name}"

        component_type = component.get("type")
        if component_type not in RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES:
            return False, f"add_component type {component_type!r} is not allowed; use one of {sorted(RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES)}"

        if component_type in RewardEditOperatorApplier.FORMULA_COMPONENT_TYPES:
            error = cls._validate_formula_component(component, structural_context)
        else:
            error = cls._validate_metric_component(component, structural_context)

        if error:
            return False, error

        return True, ""

    @classmethod
    def _validate_add_event_rule(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        rule = edit.get("event_rule")
        if not isinstance(rule, dict):
            return False, "add_event_rule requires event_rule dict"

        name = rule.get("name")
        if not name:
            return False, "event_rule requires name"

        if schema.get_component(name) or schema.get_event_rule(name):
            return False, f"reward item already exists: {name}"

        rule_type = rule.get("type")
        condition = rule.get("condition")

        if rule_type not in {"event_bonus", "event_predicate"}:
            return False, "event_rule type must be event_bonus or event_predicate"

        if not isinstance(condition, dict) or not condition:
            return False, "event_rule condition must be non-empty dict"

        if rule_type == "event_predicate":
            ast_node = condition.get("expr_ast") or condition.get("condition_ast")
            if ast_node is None:
                return False, "event_predicate requires condition.expr_ast"
            err = cls._validate_ast_text(ast_node, structural_context)
            if err:
                return False, f"event_predicate condition_ast invalid: {err}"
            if "expression" in condition or "formula" in condition:
                return False, "event_predicate string expression/formula is forbidden; use condition.expr_ast"
            return True, ""

        return True, ""

    @classmethod
    def _validate_formula_component(cls, component: Dict[str, Any], structural_context: Dict[str, Any]) -> str:
        component_type = component.get("type")
        params = component.get("params", {}) or {}
        formula_ast = component.get("formula_ast") or params.get("formula_ast")

        if formula_ast is None:
            return f"{component_type} requires formula_ast"

        err = cls._validate_ast_text(formula_ast, structural_context)
        if err:
            return f"{component_type} formula_ast invalid: {err}"

        if component_type == "conditional_formula_component":
            condition_ast = component.get("condition_ast") or params.get("condition_ast")
            if condition_ast is None:
                return "conditional_formula_component requires condition_ast"
            err = cls._validate_ast_text(condition_ast, structural_context)
            if err:
                return f"conditional_formula_component condition_ast invalid: {err}"

        if "formula" in component or "formula" in params:
            return f"{component_type} string formula is forbidden; use formula_ast"
        if "condition" in component or "condition" in params:
            return f"{component_type} string condition is forbidden; use condition_ast"

        return ""

    @classmethod
    def _validate_ast_text(cls, ast_node: Any, structural_context: Dict[str, Any]) -> str:
        allowed_vars = set(structural_context.get("allowed_formula_variables", [])) or cls.DEFAULT_ALLOWED_VARS
        validation = validate_formula_ast(ast_node, allowed_vars)
        if not validation.ok:
            return str(validation.errors)
        return ""

    @staticmethod
    def _validate_metric_component(component: Dict[str, Any], structural_context: Dict[str, Any]) -> str:
        component_type = component.get("type")
        if component_type not in RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES:
            return f"unsupported metric component type: {component_type}"

        params = component.get("params", {})
        metric = params.get("metric")

        if not metric:
            return f"{component_type} requires params.metric"

        available_metrics = set(structural_context.get("available_task_metrics", []))
        if available_metrics and metric not in available_metrics:
            return f"{component_type} references unknown task metric: {metric}"

        return ""
