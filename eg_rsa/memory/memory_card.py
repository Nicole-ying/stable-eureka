from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryCard:
    """Structured EG-RSA experience memory.

    The memory stores editable experience rather than raw reward code:
    failure mode -> attribution -> edit operator -> outcome.
    """

    memory_id: str
    env_family: str
    failure_modes: List[str]
    reward_attribution: Dict[str, Any]
    edit_plan: List[Dict[str, Any]]
    outcome: Dict[str, Any]
    lesson: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryCard":
        return cls(
            memory_id=data["memory_id"],
            env_family=data.get("env_family", "unknown"),
            failure_modes=list(data.get("failure_modes", [])),
            reward_attribution=dict(data.get("reward_attribution", {})),
            edit_plan=list(data.get("edit_plan", [])),
            outcome=dict(data.get("outcome", {})),
            lesson=data.get("lesson", ""),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "env_family": self.env_family,
            "failure_modes": self.failure_modes,
            "reward_attribution": self.reward_attribution,
            "edit_plan": self.edit_plan,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "metadata": self.metadata,
        }
