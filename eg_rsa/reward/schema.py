from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardComponent:
    """One editable reward component.

    AST-first V2:
      - formula_component requires params.formula_ast
      - conditional_formula_component requires params.condition_ast and params.formula_ast
      - string formula/condition fields are not the trusted source of execution
    """

    name: str
    type: str
    weight: float = 1.0
    inputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    clip: Optional[List[float]] = None
    enabled: bool = True
    semantic_role: Optional[str] = None
    reward_timing: Optional[str] = None
    behavior_channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardComponent":
        params = dict(data.get("params", {}) or {})

        if "formula_ast" in data and "formula_ast" not in params:
            params["formula_ast"] = data["formula_ast"]
        if "condition_ast" in data and "condition_ast" not in params:
            params["condition_ast"] = data["condition_ast"]

        return cls(
            name=data["name"],
            type=data["type"],
            weight=float(data.get("weight", 1.0)),
            inputs=list(data.get("inputs", [])),
            params=params,
            clip=data.get("clip"),
            enabled=bool(data.get("enabled", True)),
            semantic_role=data.get("semantic_role") or data.get("role"),
            reward_timing=data.get("reward_timing"),
            behavior_channel=data.get("behavior_channel"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "type": self.type,
            "weight": self.weight,
            "inputs": self.inputs,
            "params": self.params,
            "clip": self.clip,
            "enabled": self.enabled,
        }
        if self.semantic_role is not None:
            data["semantic_role"] = self.semantic_role
        if self.reward_timing is not None:
            data["reward_timing"] = self.reward_timing
        if self.behavior_channel is not None:
            data["behavior_channel"] = self.behavior_channel
        if self.type in {"formula_component", "conditional_formula_component", "action_penalty"} and "formula_ast" in self.params:
            data["formula_ast"] = self.params.get("formula_ast")
        if self.type == "conditional_formula_component" and "condition_ast" in self.params:
            data["condition_ast"] = self.params.get("condition_ast")
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class EventRule:
    """Optional event-style reward rule.

    AST-first event_predicate:
      condition.expr_ast is the trusted executable predicate.
    """

    name: str
    type: str
    weight: float = 1.0
    condition: Dict[str, Any] = field(default_factory=dict)
    one_time: bool = False
    enabled: bool = True
    semantic_role: Optional[str] = None
    reward_timing: Optional[str] = None
    behavior_channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventRule":
        raw_condition = data.get("condition", {})
        condition = dict(raw_condition or {}) if isinstance(raw_condition, dict) else {}

        if "condition_ast" in data and "expr_ast" not in condition:
            condition["expr_ast"] = data["condition_ast"]
        if "expr_ast" in data and "expr_ast" not in condition:
            condition["expr_ast"] = data["expr_ast"]
        if "duration_steps" in data and "duration_steps" not in condition:
            condition["duration_steps"] = data["duration_steps"]

        return cls(
            name=data["name"],
            type=data["type"],
            weight=float(data.get("weight", data.get("reward", 1.0))),
            condition=condition,
            one_time=bool(data.get("one_time", False)),
            enabled=bool(data.get("enabled", True)),
            semantic_role=data.get("semantic_role") or data.get("role"),
            reward_timing=data.get("reward_timing"),
            behavior_channel=data.get("behavior_channel"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "type": self.type,
            "weight": self.weight,
            "condition": self.condition,
            "one_time": self.one_time,
            "enabled": self.enabled,
        }
        if self.type == "event_predicate" and "expr_ast" in self.condition:
            data["condition_ast"] = self.condition.get("expr_ast")
        if self.semantic_role is not None:
            data["semantic_role"] = self.semantic_role
        if self.reward_timing is not None:
            data["reward_timing"] = self.reward_timing
        if self.behavior_channel is not None:
            data["behavior_channel"] = self.behavior_channel
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class RewardSchema:
    """Editable reward schema used by EG-RSA."""

    version: int = 0
    components: List[RewardComponent] = field(default_factory=list)
    event_rules: List[EventRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardSchema":
        return cls(
            version=int(data.get("version", 0)),
            components=[RewardComponent.from_dict(x) for x in data.get("components", [])],
            event_rules=[EventRule.from_dict(x) for x in data.get("event_rules", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "components": [x.to_dict() for x in self.components],
            "event_rules": [x.to_dict() for x in self.event_rules],
            "metadata": self.metadata,
        }

    def get_component(self, name: str) -> Optional[RewardComponent]:
        for component in self.components:
            if component.name == name:
                return component
        return None

    def get_event_rule(self, name: str) -> Optional[EventRule]:
        for rule in self.event_rules:
            if rule.name == name:
                return rule
        return None
