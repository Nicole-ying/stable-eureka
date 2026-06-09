from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


VALID_ACTIONS = {
    "apply_local_edit",
    "apply_structural_edit",
    "free_rewrite_schema",
    "continue_training",
    "evaluate_more",
    "rollback_replan",
    "run_tool",
    "early_stop",
}


@dataclass
class AgentActionDecision:
    """Unified v1 agent action.

    v0 mostly asks the LLM to output an edit_plan.
    v1 lets the agent choose the next search action first.
    """

    action: str
    confidence: float = 0.0
    reason_summary: str = ""
    tools_to_call: List[Dict[str, Any]] = field(default_factory=list)
    edit_intent: Dict[str, Any] = field(default_factory=dict)
    safety_requirements: List[str] = field(default_factory=list)
    memory_items_to_write: List[Dict[str, Any]] = field(default_factory=list)
    fallback_action: str = "continue_training"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_action(self) -> str:
        if self.action in VALID_ACTIONS:
            return self.action
        return "run_tool" if self.tools_to_call else "continue_training"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.normalized_action(),
            "raw_action": self.action,
            "confidence": float(self.confidence),
            "reason_summary": self.reason_summary,
            "tools_to_call": self.tools_to_call,
            "edit_intent": self.edit_intent,
            "safety_requirements": self.safety_requirements,
            "memory_items_to_write": self.memory_items_to_write,
            "fallback_action": self.fallback_action,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentActionDecision":
        data = data or {}
        return cls(
            action=str(data.get("action", "continue_training")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reason_summary=str(data.get("reason_summary", "")),
            tools_to_call=list(data.get("tools_to_call", []) or []),
            edit_intent=dict(data.get("edit_intent", {}) or {}),
            safety_requirements=list(data.get("safety_requirements", []) or []),
            memory_items_to_write=list(data.get("memory_items_to_write", []) or []),
            fallback_action=str(data.get("fallback_action", "continue_training")),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def continue_training(cls, reason: str, confidence: float = 0.7) -> "AgentActionDecision":
        return cls(
            action="continue_training",
            confidence=confidence,
            reason_summary=reason,
        )

    @classmethod
    def run_tool(cls, tool_name: str, args: Dict[str, Any], reason: str) -> "AgentActionDecision":
        return cls(
            action="run_tool",
            confidence=0.7,
            reason_summary=reason,
            tools_to_call=[{"tool_name": tool_name, "args": args}],
        )
