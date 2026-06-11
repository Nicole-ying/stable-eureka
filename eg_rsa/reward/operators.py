from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List

from eg_rsa.reward.schema import EventRule, RewardComponent, RewardSchema


class RewardEditOperatorApplier:
    """Apply constrained AST-first edit plans to RewardSchema.

    LLMs output edit instructions and AST objects, never Python code or formula strings.
    """

    ALLOWED_OPERATORS = {
        "increase_weight",
        "decrease_weight",
        "clip_component",
        "disable_component",
        "add_component",
        "add_event_rule",
        "convert_to_one_time_event",
        "add_duration_condition",
        "replace_formula",
        "replace_condition",
        "add_formula_component",
        "add_conditional_formula_component",
        "add_action_penalty",
        "add_event_predicate",
    }

    METRIC_COMPONENT_TYPES = {
        "metric_value",
        "metric_delta",
        "metric_threshold_bonus",
        "metric_stagnation_penalty",
    }

    FORMULA_COMPONENT_TYPES = {
        "formula_component",
        "conditional_formula_component",
        "action_penalty",
    }

    ADDABLE_COMPONENT_TYPES = set(METRIC_COMPONENT_TYPES) | set(FORMULA_COMPONENT_TYPES)

    @classmethod
    def apply(cls, schema: RewardSchema, edit_plan: Iterable[Dict[str, Any]]) -> RewardSchema:
        new_schema = copy.deepcopy(schema)
        for edit in edit_plan:
            op = edit.get("operator") or edit.get("op")
            if op not in cls.ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported edit operator: {op}")
            getattr(cls, f"_{op}")(new_schema, edit)
        new_schema.version += 1
        return new_schema

    @staticmethod
    def allowed_operator_descriptions() -> List[Dict[str, Any]]:
        return [
            {"operator": "increase_weight", "required": ["target", "factor"], "description": "Multiply existing reward item weight by factor > 1."},
            {"operator": "decrease_weight", "required": ["target", "factor"], "description": "Multiply existing reward item weight by 0 < factor < 1."},
            {"operator": "clip_component", "required": ["target", "clip"], "description": "Set existing component clip range [min, max]."},
            {"operator": "disable_component", "required": ["target"], "description": "Disable harmful/redundant component or event rule."},
            {"operator": "replace_formula", "required": ["target", "formula_ast"], "description": "Replace params.formula_ast of an existing AST formula component."},
            {"operator": "replace_condition", "required": ["target", "condition_ast"], "description": "Replace params.condition_ast or event condition.expr_ast."},
            {"operator": "add_formula_component", "required": ["component.formula_ast"], "description": "Add formula_component using AST IR."},
            {"operator": "add_conditional_formula_component", "required": ["component.condition_ast", "component.formula_ast"], "description": "Add conditional_formula_component using AST IR."},
            {"operator": "add_action_penalty", "required": ["component.formula_ast"], "description": "Alias for AST formula_component with semantic_role=control_cost."},
            {"operator": "add_event_predicate", "required": ["event_rule.condition.expr_ast"], "description": "Add event_predicate using AST boolean predicate."},
            {"operator": "add_component", "required": ["component"], "description": "Backward-compatible structural add; AST fields are still required for formula types."},
            {"operator": "add_event_rule", "required": ["event_rule"], "description": "Backward-compatible event add; event_predicate requires condition.expr_ast."},
            {"operator": "convert_to_one_time_event", "required": ["target"], "description": "Convert event rule or event_bonus component into one-time event."},
            {"operator": "add_duration_condition", "required": ["target", "duration_steps"], "description": "Set event rule duration requirement."},
        ]

    @staticmethod
    def _find_component_or_rule(schema: RewardSchema, target: str):
        obj = schema.get_component(target)
        if obj is not None:
            return obj
        obj = schema.get_event_rule(target)
        if obj is not None:
            return obj
        raise ValueError(f"Target not found in reward schema: {target}")

    @classmethod
    def _increase_weight(cls, schema: RewardSchema, edit: Dict[str, Any]) -> None:
        obj = cls._find_component_or_rule(schema, edit["target"])
        factor = float(edit["factor"])
        if factor <= 1.0:
            raise ValueError("increase_weight requires factor > 1")
        obj.weight *= factor

    @classmethod
    def _decrease_weight(cls, schema: RewardSchema, edit: Dict[str, Any]) -> None:
        obj = cls._find_component_or_rule(schema, edit["target"])
        factor = float(edit["factor"])
        if not 0.0 < factor < 1.0:
            raise ValueError("decrease_weight requires 0 < factor < 1")
        obj.weight *= factor

    @staticmethod
    def _clip_component(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        component = schema.get_component(edit["target"])
        if component is None:
            raise ValueError("clip_component target must be a component")
        clip = edit["clip"]
        if len(clip) != 2 or float(clip[0]) > float(clip[1]):
            raise ValueError(f"Invalid clip range: {clip}")
        component.clip = [float(clip[0]), float(clip[1])]

    @classmethod
    def _disable_component(cls, schema: RewardSchema, edit: Dict[str, Any]) -> None:
        obj = cls._find_component_or_rule(schema, edit["target"])
        obj.enabled = False

    @staticmethod
    def _replace_formula(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        target = edit["target"]
        formula_ast = edit.get("formula_ast")
        if formula_ast is None:
            raise ValueError("replace_formula requires formula_ast")
        component = schema.get_component(target)
        if component is None:
            raise ValueError("replace_formula target must be a component")
        if component.type not in RewardEditOperatorApplier.FORMULA_COMPONENT_TYPES:
            raise ValueError(f"replace_formula target must be formula-native component, got {component.type}")
        component.params = dict(component.params or {})
        component.params["formula_ast"] = formula_ast

    @staticmethod
    def _replace_condition(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        target = edit["target"]
        condition_ast = edit.get("condition_ast")
        if condition_ast is None and isinstance(edit.get("condition"), dict):
            condition_ast = edit["condition"].get("expr_ast") or edit["condition"].get("condition_ast")
        if condition_ast is None:
            raise ValueError("replace_condition requires condition_ast")

        component = schema.get_component(target)
        if component is not None:
            if component.type != "conditional_formula_component":
                raise ValueError("replace_condition component target must be conditional_formula_component")
            component.params = dict(component.params or {})
            component.params["condition_ast"] = condition_ast
            return

        rule = schema.get_event_rule(target)
        if rule is not None:
            if rule.type != "event_predicate":
                raise ValueError("replace_condition event target must be event_predicate")
            old = dict(rule.condition or {})
            old["expr_ast"] = condition_ast
            rule.condition = old
            return

        raise ValueError(f"replace_condition target not found: {target}")

    @staticmethod
    def _add_component(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        component = RewardComponent.from_dict(edit["component"])
        if component.type not in RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES:
            raise ValueError(
                f"Unsupported add_component type: {component.type}. "
                f"Supported: {sorted(RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES)}"
            )
        if schema.get_component(component.name) or schema.get_event_rule(component.name):
            raise ValueError(f"Reward item already exists: {component.name}")
        schema.components.append(component)

    @staticmethod
    def _add_formula_component(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        component = dict(edit["component"])
        component["type"] = "formula_component"
        RewardEditOperatorApplier._add_component(schema, {"component": component})

    @staticmethod
    def _add_conditional_formula_component(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        component = dict(edit["component"])
        component["type"] = "conditional_formula_component"
        RewardEditOperatorApplier._add_component(schema, {"component": component})

    @staticmethod
    def _add_action_penalty(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        component = dict(edit["component"])
        component["type"] = "formula_component"
        component.setdefault("semantic_role", "control_cost")
        component.setdefault("reward_timing", "dense")
        component.setdefault("behavior_channel", "control")
        RewardEditOperatorApplier._add_component(schema, {"component": component})

    @staticmethod
    def _add_event_rule(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        rule = EventRule.from_dict(edit["event_rule"])
        if schema.get_component(rule.name) or schema.get_event_rule(rule.name):
            raise ValueError(f"Reward item already exists: {rule.name}")
        if rule.type not in {"event_bonus", "event_predicate"}:
            raise ValueError("add_event_rule supports event_bonus or event_predicate rules")
        if not rule.condition:
            raise ValueError("event_rule condition cannot be empty")
        schema.event_rules.append(rule)

    @staticmethod
    def _add_event_predicate(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        rule = dict(edit["event_rule"])
        rule["type"] = "event_predicate"
        RewardEditOperatorApplier._add_event_rule(schema, {"event_rule": rule})

    @staticmethod
    def _convert_to_one_time_event(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        target = edit["target"]
        rule = schema.get_event_rule(target)
        if rule is not None:
            rule.one_time = True
            return
        component = schema.get_component(target)
        if component is None:
            raise ValueError(f"Target not found: {target}")
        if component.type != "event_bonus":
            raise ValueError("convert_to_one_time_event requires event_bonus component or event rule")
        event_name = component.params.get("event", component.name)
        schema.event_rules.append(
            EventRule(
                name=f"{component.name}_one_time",
                type="event_bonus",
                weight=component.weight,
                condition={event_name: True},
                one_time=True,
                enabled=True,
            )
        )
        component.enabled = False

    @staticmethod
    def _add_duration_condition(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        rule = schema.get_event_rule(edit["target"])
        if rule is None:
            raise ValueError("add_duration_condition target must be an event rule")
        duration_steps = int(edit["duration_steps"])
        if duration_steps <= 0:
            raise ValueError("duration_steps must be positive")
        rule.condition["duration_steps"] = duration_steps
