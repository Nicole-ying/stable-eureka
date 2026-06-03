from __future__ import annotations

from typing import Any, Dict, List, Optional

from eg_rsa.llm.edit_prompt import build_edit_prompt
from eg_rsa.llm.json_parser import extract_json_object


class EditAgent:
    """Generate constrained reward edit plans.

    This class is intentionally pluggable. If an LLM client is provided, it asks
    the model to output JSON. Otherwise it falls back to a deterministic rule
    policy so the real training loop remains runnable.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def generate_edit_plan(
        self,
        task_description: str,
        current_reward_schema: Dict[str, Any],
        diagnostic_report: Dict[str, Any],
        retrieved_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.llm_client is None:
            return self._fallback_response(diagnostic_report)

        prompt = build_edit_prompt(
            task_description=task_description,
            current_reward_schema=current_reward_schema,
            diagnostic_report=diagnostic_report,
            retrieved_memories=retrieved_memories,
        )
        response_text = self.llm_client.generate(prompt)
        parsed = extract_json_object(response_text)
        if "edit_plan" not in parsed or not isinstance(parsed["edit_plan"], list):
            raise ValueError("LLM edit response must contain a list field named edit_plan")
        return parsed

    @staticmethod
    def _fallback_response(diagnostic_report: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics = diagnostic_report.get("diagnostics", diagnostic_report)
        target = diagnostics.get("dominant_component")
        modes = set(diagnostics.get("failure_modes", []))
        edits: List[Dict[str, Any]] = []
        if target and "single_component_dominance" in modes:
            edits.append({"operator": "decrease_weight", "target": target, "factor": 0.5})
        if target and "repeated_event_exploitation" in modes:
            edits.append({"operator": "convert_to_one_time_event", "target": target})
        if not edits and target:
            edits.append({"operator": "clip_component", "target": target, "clip": [-1.0, 1.0]})
        return {
            "diagnosis": "Fallback edit policy generated a conservative edit plan from diagnostics.",
            "edit_plan": edits,
        }
