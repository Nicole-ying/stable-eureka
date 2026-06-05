from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eg_rsa.reward.operators import RewardEditOperatorApplier
from eg_rsa.reward.schema import RewardSchema


@dataclass
class EditPlanValidationResult:
    valid_edits: List[Dict[str, Any]] = field(default_factory=list)
    rejected_edits: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.valid_edits) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"is_valid": self.is_valid, "valid_edits": self.valid_edits, "rejected_edits": self.rejected_edits, "errors": self.errors}


class EditPlanValidator:
    """Validate LLM/fallback edit plans before applying operators."""

    @classmethod
    def validate(
        cls,
        schema: RewardSchema,
        edit_plan: List[Dict[str, Any]],
        structural_context: Optional[Dict[str, Any]] = None,
    ) -> EditPlanValidationResult:
        result = EditPlanValidationResult()
        if not isinstance(edit_plan, list):
            result.errors.append("edit_plan must be a list")
            return result
        for idx, edit in enumerate(edit_plan):
            ok, error = cls._validate_one(schema, edit, structural_context or {})
            if ok:
                result.valid_edits.append(edit)
            else:
                rejected = dict(edit) if isinstance(edit, dict) else {"raw_edit": edit}
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
    def _validate_one(cls, schema: RewardSchema, edit: Any, structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(edit, dict):
            return False, "edit must be a dict"
        op = edit.get("operator") or edit.get("op")
        if op not in RewardEditOperatorApplier.ALLOWED_OPERATORS:
            return False, f"unsupported operator: {op}"
        if op in {"increase_weight", "decrease_weight", "clip_component", "disable_component", "convert_to_one_time_event", "add_duration_condition", "reshape_sparse_to_dense"}:
            target = edit.get("target")
            if not target:
                return False, f"{op} requires target"
            if schema.get_component(target) is None and schema.get_event_rule(target) is None:
                return False, f"target not found: {target}"
        if op == "increase_weight":
            factor = float(edit.get("factor", 0.0))
            if factor <= 1.0:
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
        if op == "convert_to_one_time_event":
            target = edit.get("target")
            component = schema.get_component(target)
            rule = schema.get_event_rule(target)
            if component is not None and component.type != "event_bonus":
                return False, "convert_to_one_time_event requires an event_bonus component or event rule"
            if rule is None and component is None:
                return False, f"target not found: {target}"
        if op == "add_component":
            component = edit.get("component")
            if not isinstance(component, dict):
                return False, "add_component requires component dict"
            name = component.get("name")
            if not name:
                return False, "new component requires name"
            if schema.get_component(name) or schema.get_event_rule(name):
                return False, f"reward item already exists: {name}"
        if op == "add_event_rule":
            rule = edit.get("event_rule")
            if not isinstance(rule, dict):
                return False, "add_event_rule requires event_rule dict"
            name = rule.get("name")
            if not name:
                return False, "event_rule requires name"
            if schema.get_component(name) or schema.get_event_rule(name):
                return False, f"reward item already exists: {name}"
            if rule.get("type") != "event_bonus":
                return False, "event_rule type must be event_bonus"
            condition = rule.get("condition")
            if not isinstance(condition, dict) or not condition:
                return False, "event_rule condition must be non-empty dict"
            unknown_events = cls._unknown_condition_events(condition, structural_context)
            if unknown_events:
                return False, f"event_rule condition references unknown events: {unknown_events}"
        if op == "add_duration_condition":
            if schema.get_event_rule(edit.get("target")) is None:
                return False, "add_duration_condition target must be an event rule"
            if int(edit.get("duration_steps", 0)) <= 0:
                return False, "duration_steps must be positive"
        return True, ""

    @staticmethod
    def _unknown_condition_events(condition: Dict[str, Any], structural_context: Dict[str, Any]) -> List[str]:
        available_events = set(structural_context.get("available_events", []))
        if not available_events:
            return []
        unknown = []
        for key in condition.keys():
            if key == "duration_steps":
                continue
            if key not in available_events:
                unknown.append(key)
        return unknown
