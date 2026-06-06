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
        retrieved_lessons: Optional[List[Dict[str, Any]]] = None,
        reflection_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.llm_client is None:
            return self._fallback_response(diagnostic_report, current_reward_schema, reflection_report or {})

        prompt = build_edit_prompt(
            task_description=task_description,
            current_reward_schema=current_reward_schema,
            diagnostic_report=diagnostic_report,
            retrieved_memories=retrieved_memories,
            retrieved_lessons=retrieved_lessons or [],
            reflection_report=reflection_report or {},
        )
        response_text = self.llm_client.generate(prompt)
        parsed = extract_json_object(response_text)
        if "edit_plan" not in parsed or not isinstance(parsed["edit_plan"], list):
            raise ValueError("LLM edit response must contain a list field named edit_plan")
        return self._normalize_response(parsed, reflection_report or {})

    @staticmethod
    def _normalize_response(parsed: Dict[str, Any], reflection_report: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(parsed or {})
        editor = dict(parsed.get("reward_editor", {}) or {})
        strategy = dict(reflection_report.get("strategy", {}) or {})
        plan_type = editor.get("plan_type") or parsed.get("plan_type") or strategy.get("plan_type") or "single_edit"
        atomicity = editor.get("atomicity") or parsed.get("atomicity") or strategy.get("atomicity") or "separable"
        max_reasonable_edits = editor.get("max_reasonable_edits") or parsed.get("max_reasonable_edits") or strategy.get("max_reasonable_edits") or 1
        editor["plan_type"] = plan_type
        editor["atomicity"] = atomicity
        editor["max_reasonable_edits"] = int(max_reasonable_edits)
        parsed["reward_editor"] = editor
        parsed["plan_type"] = plan_type
        parsed["atomicity"] = atomicity
        parsed["max_reasonable_edits"] = int(max_reasonable_edits)
        return parsed

    @staticmethod
    def _fallback_response(diagnostic_report: Dict[str, Any], reward_schema: Dict[str, Any], reflection_report: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics = diagnostic_report.get("diagnostics", diagnostic_report)
        target = diagnostics.get("dominant_component")
        modes = set(diagnostics.get("failure_modes", []))
        target_info = EditAgent._find_reward_item(reward_schema, target)
        strategy = dict(reflection_report.get("strategy", {}) or {})
        plan_type = strategy.get("plan_type", "single_edit")
        atomicity = strategy.get("atomicity", "separable")
        max_reasonable_edits = int(strategy.get("max_reasonable_edits", 1) or 1)
        edits: List[Dict[str, Any]] = []

        if target and target_info is not None:
            item_kind = target_info.get("kind")
            item_type = target_info.get("type")
            is_event_like = item_kind == "event_rule" or item_type == "event_bonus"

            if "single_component_dominance" in modes:
                edits.append({"operator": "decrease_weight", "target": target, "factor": 0.5})
            elif "repeated_event_exploitation" in modes and is_event_like:
                edits.append({"operator": "convert_to_one_time_event", "target": target})
            elif item_kind == "component":
                edits.append({"operator": "clip_component", "target": target, "clip": [-1.0, 1.0]})

        return {
            "diagnosis": "Fallback edit policy generated a schema-aware conservative edit plan from diagnostics and reflection.",
            "reward_editor": {
                "edit_decision": "edit" if edits else "no_edit",
                "next_action": "apply_edit" if edits else "early_stop",
                "plan_type": plan_type,
                "atomicity": atomicity,
                "max_reasonable_edits": max_reasonable_edits,
                "rationale": "Fallback policy follows the reflection strategy when available.",
                "edit_plan": edits,
            },
            "plan_type": plan_type,
            "atomicity": atomicity,
            "max_reasonable_edits": max_reasonable_edits,
            "edit_plan": edits,
        }

    @staticmethod
    def _find_reward_item(reward_schema: Dict[str, Any], target: Optional[str]) -> Optional[Dict[str, Any]]:
        if not target:
            return None
        for component in reward_schema.get("components", []):
            if component.get("name") == target:
                item = dict(component)
                item["kind"] = "component"
                return item
        for rule in reward_schema.get("event_rules", []):
            if rule.get("name") == target:
                item = dict(rule)
                item["kind"] = "event_rule"
                return item
        return None
