from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardComponent:
    """One editable reward component.

    v1 adds semantic metadata so tools can reason over reward roles rather than
    environment-specific component names. These metadata fields do not affect
    reward execution; they are used by diagnostics, memory, and repair tools.
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
        return cls(
            name=data["name"],
            type=data["type"],
            weight=float(data.get("weight", 1.0)),
            inputs=list(data.get("inputs", [])),
            params=dict(data.get("params", {})),
            clip=data.get("clip"),
            enabled=bool(data.get("enabled", True)),
            semantic_role=data.get("semantic_role"),
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
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class EventRule:
    """Optional event-style reward rule.

    Event rules are useful for one-time bonuses and duration-conditioned bonuses.
    v1 also preserves semantic metadata for role-based diagnostics.
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
        return cls(
            name=data["name"],
            type=data["type"],
            weight=float(data.get("weight", 1.0)),
            condition=dict(data.get("condition", {})),
            one_time=bool(data.get("one_time", False)),
            enabled=bool(data.get("enabled", True)),
            semantic_role=data.get("semantic_role"),
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
