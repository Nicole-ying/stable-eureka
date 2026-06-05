from __future__ import annotations

from typing import Any, Dict, List, Optional

from eg_rsa.llm.json_parser import extract_json_object
from eg_rsa.llm.structural_search_prompt import build_structural_search_prompt


class StructuralSearchAgent:
    """Generate structural reward edits when local editing is insufficient."""

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
        return parsed

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
