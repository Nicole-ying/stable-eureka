from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardComponent:
    """One editable reward component.

    The LLM should not emit Python reward code directly.  It should emit or edit
    this schema.  The compiler then turns the schema into executable Python.
    """

    name: str
    type: str
    weight: float = 1.0
    inputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    clip: Optional[List[float]] = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardComponent":
        return cls(
            name=data["name"],
            type=data["type"],
            weight=float(data.get("weight", 1.0)),
            inputs=list(data.get("inputs", [])),
            params=dict(data.get("params", {})),
            clip=data.get("clip"),
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "weight": self.weight,
            "inputs": self.inputs,
            "params": self.params,
            "clip": self.clip,
            "enabled": self.enabled,
        }


@dataclass
class EventRule:
    """Optional event-style reward rule.

    Event rules are useful for one-time bonuses and duration-conditioned bonuses.
    They are separated from normal dense components to reduce repeated-event
    reward hacking.
    """

    name: str
    type: str
    weight: float = 1.0
    condition: Dict[str, Any] = field(default_factory=dict)
    one_time: bool = False
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventRule":
        return cls(
            name=data["name"],
            type=data["type"],
            weight=float(data.get("weight", 1.0)),
            condition=dict(data.get("condition", {})),
            one_time=bool(data.get("one_time", False)),
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "weight": self.weight,
            "condition": self.condition,
            "one_time": self.one_time,
            "enabled": self.enabled,
        }


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
