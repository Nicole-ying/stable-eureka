from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from eg_rsa.llm.json_parser import extract_json_object
from eg_rsa.llm.structural_search_prompt import build_structural_search_prompt


class StructuralSearchAgent:
    """Generate structural reward edits when local editing is insufficient.

    The LLM may produce a compact event-rule form such as
    {"event": "success", "weight": 100}.  Before validation, this agent
    normalizes such compact edits into the trusted RewardSchema EventRule
    format.  The normalization is schema-level only and does not encode any
    environment-specific policy.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def generate_structural_edit(
        self,
        task_description: str,
        current_reward_schema: Dict[str, Any],
        diagnostic_report: Dict[str, Any],
        retrieved_lessons: List[Dict[str, Any]],
        structural_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.llm_client is None:
            return self._fallback_response(structural_context)
        prompt = build_structural_search_prompt(
            task_description=task_description,
            current_reward_schema=current_reward_schema,
            diagnostic_report=diagnostic_report,
            retrieved_lessons=retrieved_lessons,
            structural_context=structural_context,
        )
        response_text = self.llm_client.generate(prompt)
        parsed = extract_json_object(response_text)
        if "edit_plan" not in parsed or not isinstance(parsed["edit_plan"], list):
            raise ValueError("Structural search response must contain list field edit_plan")
        return self._normalize_response(parsed, structural_context)

    @classmethod
    def _normalize_response(cls, response: Dict[str, Any], structural_context: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(response)
        normalized_plan = []
        normalization_notes = []
        for edit in response.get("edit_plan", []):
            normalized_edit, notes = cls._normalize_edit(edit, structural_context)
            normalized_plan.append(normalized_edit)
            normalization_notes.extend(notes)
        normalized["edit_plan"] = normalized_plan

        editor = normalized.get("reward_editor")
        if isinstance(editor, dict) and isinstance(editor.get("edit_plan"), list):
            editor_plan = []
            for edit in editor.get("edit_plan", []):
                normalized_edit, notes = cls._normalize_edit(edit, structural_context)
                editor_plan.append(normalized_edit)
                normalization_notes.extend(notes)
            editor["edit_plan"] = editor_plan
            normalized["reward_editor"] = editor

        if normalization_notes:
            normalized["normalization_notes"] = normalization_notes
        return normalized

    @classmethod
    def _normalize_edit(cls, edit: Dict[str, Any], structural_context: Dict[str, Any]):
        if not isinstance(edit, dict):
            return edit, ["Skipped non-dict edit during structural normalization."]
        if edit.get("operator") != "add_event_rule":
            return edit, []
        rule = edit.get("event_rule")
        if not isinstance(rule, dict):
            return edit, ["add_event_rule edit has no event_rule dict; left unchanged."]

        if cls._is_complete_event_rule(rule):
            cleaned = dict(edit)
            cleaned["event_rule"] = cls._clean_complete_rule(rule)
            return cleaned, []

        event_name = rule.get("event") or rule.get("event_name") or rule.get("metric")
        available_events = set(structural_context.get("available_events", []))
        if not event_name or (available_events and event_name not in available_events):
            return edit, [f"Cannot normalize add_event_rule because event {event_name!r} is not available."]

        duration = rule.get("duration_steps", None)
        condition = {event_name: True}
        if isinstance(duration, int) and duration > 1:
            condition["duration_steps"] = duration
        elif isinstance(duration, float) and duration > 1:
            condition["duration_steps"] = int(duration)

        normalized_rule = {
            "name": rule.get("name") or cls._safe_rule_name(event_name, bool(rule.get("one_time", True))),
            "type": rule.get("type") or "event_bonus",
            "weight": float(rule.get("weight", 20.0)),
            "condition": condition,
            "one_time": bool(rule.get("one_time", True)),
            "enabled": bool(rule.get("enabled", True)),
        }
        return {"operator": "add_event_rule", "event_rule": normalized_rule}, [
            f"Normalized compact add_event_rule for event {event_name!r} into EventRule schema."
        ]

    @staticmethod
    def _is_complete_event_rule(rule: Dict[str, Any]) -> bool:
        return all(key in rule for key in ["name", "type", "weight", "condition", "one_time", "enabled"])

    @staticmethod
    def _clean_complete_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(rule)
        cleaned["weight"] = float(cleaned.get("weight", 0.0))
        cleaned["one_time"] = bool(cleaned.get("one_time", False))
        cleaned["enabled"] = bool(cleaned.get("enabled", True))
        return cleaned

    @staticmethod
    def _safe_rule_name(event_name: str, one_time: bool) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(event_name)).strip("_")
        suffix = "once" if one_time else "event"
        return f"r_{safe}_{suffix}"

    @staticmethod
    def _fallback_response(structural_context: Dict[str, Any]) -> Dict[str, Any]:
        events = structural_context.get("available_events", [])
        preferred = structural_context.get("preferred_success_events", [])
        candidate_event = preferred[0] if preferred else (events[0] if events else None)
        if not candidate_event:
            return {
                "diagnosis": "No configured event is available for structural search.",
                "edit_plan": [],
                "reward_editor": {"edit_decision": "no_edit", "next_action": "early_stop"},
            }
        return {
            "structural_analysis": {
                "missing_signal_hypothesis": "A non-repeatable positive completion signal may be missing.",
                "why_local_edit_is_insufficient": "Local edits cannot create a new gated event rule.",
                "memory_constraints": [],
                "safety_constraints": ["Use one_time to avoid repeated-event exploitation."],
            },
            "reward_editor": {
                "edit_decision": "edit",
                "next_action": "apply_edit",
                "rationale": "Fallback structural search adds a one-time event rule using a configured event.",
                "edit_plan": [
                    {
                        "operator": "add_event_rule",
                        "event_rule": {
                            "name": f"r_structural_{candidate_event}",
                            "type": "event_bonus",
                            "weight": 20.0,
                            "condition": {candidate_event: True, "duration_steps": 3},
                            "one_time": True,
                            "enabled": True,
                        },
                    }
                ],
            },
            "auditor_check": {"approved": True, "issues": [], "final_action": "apply_edit"},
            "diagnosis": "Fallback structural search proposed a one-time configured event rule.",
            "edit_plan": [
                {
                    "operator": "add_event_rule",
                    "event_rule": {
                        "name": f"r_structural_{candidate_event}",
                        "type": "event_bonus",
                        "weight": 20.0,
                        "condition": {candidate_event: True, "duration_steps": 3},
                        "one_time": True,
                        "enabled": True,
                    },
                }
            ],
        }
