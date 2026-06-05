from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List

from eg_rsa.reward.schema import EventRule, RewardComponent, RewardSchema


class RewardEditOperatorApplier:
    """Apply a constrained edit plan to a RewardSchema.

    LLM outputs must be edit instructions, not executable code.  This class is
    the trusted executor. Unsupported operators fail fast.
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
        "reshape_sparse_to_dense",
    }

    METRIC_COMPONENT_TYPES = {
        "metric_value",
        "metric_delta",
        "metric_threshold_bonus",
        "metric_stagnation_penalty",
    }

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
            {"operator": "increase_weight", "required": ["target", "factor"], "description": "Multiply a component or event rule weight by factor > 1."},
            {"operator": "decrease_weight", "required": ["target", "factor"], "description": "Multiply a component or event rule weight by 0 < factor < 1."},
            {"operator": "clip_component", "required": ["target", "clip"], "description": "Set component clip range [min, max]."},
            {"operator": "disable_component", "required": ["target"], "description": "Disable a harmful or redundant component."},
            {
                "operator": "add_component",
                "required": ["component"],
                "description": "Add a new dense or metric-based component following RewardComponent schema. Metric component types may include metric_value, metric_delta, metric_threshold_bonus, metric_stagnation_penalty and must reference configured task_metrics via params.metric.",
            },
            {"operator": "add_event_rule", "required": ["event_rule"], "description": "Add a gated event reward rule. The rule condition must reference available event flags; supports one_time and duration_steps."},
            {"operator": "convert_to_one_time_event", "required": ["target"], "description": "Convert an event rule or event_bonus component into one-time reward to reduce repeated event exploitation."},
            {"operator": "add_duration_condition", "required": ["target", "duration_steps"], "description": "Require an event rule to remain true for K steps."},
            {"operator": "reshape_sparse_to_dense", "required": ["target", "new_type"], "description": "Change a sparse event-like component into a dense shaping component type."},
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
    def _add_component(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        component = RewardComponent.from_dict(edit["component"])
        if schema.get_component(component.name) or schema.get_event_rule(component.name):
            raise ValueError(f"Reward item already exists: {component.name}")
        schema.components.append(component)

    @staticmethod
    def _add_event_rule(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        rule = EventRule.from_dict(edit["event_rule"])
        if schema.get_component(rule.name) or schema.get_event_rule(rule.name):
            raise ValueError(f"Reward item already exists: {rule.name}")
        if rule.type != "event_bonus":
            raise ValueError("add_event_rule currently supports event_bonus rules only")
        if not rule.condition:
            raise ValueError("event_rule condition cannot be empty")
        schema.event_rules.append(rule)

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
        schema.event_rules.append(EventRule(name=f"{component.name}_one_time", type="event_bonus", weight=component.weight, condition={event_name: True}, one_time=True, enabled=True))
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

    @staticmethod
    def _reshape_sparse_to_dense(schema: RewardSchema, edit: Dict[str, Any]) -> None:
        component = schema.get_component(edit["target"])
        if component is None:
            raise ValueError("reshape_sparse_to_dense target must be a component")
        component.type = edit["new_type"]
        if "params" in edit:
            component.params.update(edit["params"])
        component.enabled = True
