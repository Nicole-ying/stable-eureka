from __future__ import annotations

from typing import Any, Dict, List, Optional

from eg_rsa.llm.json_parser import extract_json_object
from eg_rsa.llm.reflection_prompt import build_reflection_prompt


class ReflectionAgent:
    """Reflect on diagnostics and memory before reward editing.

    This is a separate LLM agent from the reward editor. It does not output an
    executable edit plan. It outputs a strategy report that the editor and gate
    should preserve, especially whether a plan is atomic or separable.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def reflect(
        self,
        task_description: str,
        current_reward_schema: Dict[str, Any],
        diagnostic_report: Dict[str, Any],
        retrieved_memories: List[Dict[str, Any]],
        retrieved_lessons: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if self.llm_client is None:
            return self._fallback_reflection(diagnostic_report, retrieved_lessons or [])
        prompt = build_reflection_prompt(
            task_description=task_description,
            current_reward_schema=current_reward_schema,
            diagnostic_report=diagnostic_report,
            retrieved_memories=retrieved_memories,
            retrieved_lessons=retrieved_lessons or [],
        )
        response_text = self.llm_client.generate(prompt)
        parsed = extract_json_object(response_text)
        return self._normalize(parsed)

    @staticmethod
    def _normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
        parsed = dict(parsed or {})
        strategy = dict(parsed.get("strategy", {}) or {})
        strategy.setdefault("recommended_next_action", "apply_edit")
        strategy.setdefault("plan_type", "single_edit")
        strategy.setdefault("atomicity", "separable")
        strategy.setdefault("max_reasonable_edits", 1)
        strategy.setdefault("editor_constraints", [])
        strategy.setdefault("must_preserve", [])
        strategy.setdefault("must_avoid", [])
        parsed["strategy"] = strategy
        parsed.setdefault("reflection_summary", "ReflectionAgent produced a normalized reflection report.")
        parsed.setdefault("failure_assessment", {})
        parsed.setdefault("memory_assessment", {})
        parsed.setdefault("auditor_hints", {})
        return parsed

    @staticmethod
    def _fallback_reflection(diagnostic_report: Dict[str, Any], retrieved_lessons: List[Dict[str, Any]]) -> Dict[str, Any]:
        diagnostics = diagnostic_report.get("diagnostics", diagnostic_report)
        modes = set(diagnostics.get("failure_modes", []))
        avoid_actions = []
        for lesson in retrieved_lessons:
            quality = lesson.get("quality", {}) if isinstance(lesson, dict) else {}
            if quality.get("reuse_confidence", 0.0) == 0.0:
                avoid_actions.extend(lesson.get("recommendation", {}).get("avoid_next", []))
        if "shaping_goal_mismatch" in modes:
            plan_type = "coupled_rebalancing"
            atomicity = "atomic"
            max_edits = 3
        elif "repeated_event_exploitation" in modes:
            plan_type = "single_edit"
            atomicity = "separable"
            max_edits = 1
        else:
            plan_type = "single_edit"
            atomicity = "separable"
            max_edits = 1
        return {
            "reflection_summary": "Fallback reflection selected a conservative strategy from diagnostic modes and lessons.",
            "failure_assessment": {
                "observed_facts": [],
                "likely_true_failures": list(modes),
                "likely_false_positives": [],
                "root_cause_hypotheses": [],
                "failure_kind": "mixed" if len(modes) > 1 else (next(iter(modes)) if modes else "unclear"),
                "confidence": 0.5,
            },
            "memory_assessment": {
                "reusable_lessons": [],
                "failed_or_weak_lessons": [],
                "conflicting_lessons": [],
                "avoid_actions": avoid_actions,
                "recommended_actions": [],
                "memory_confidence": 0.3,
            },
            "strategy": {
                "recommended_next_action": "apply_edit",
                "plan_type": plan_type,
                "atomicity": atomicity,
                "why_atomic_or_separable": "Fallback uses atomic coupled rebalancing when shaping-goal mismatch is present.",
                "max_reasonable_edits": max_edits,
                "editor_constraints": [],
                "must_preserve": [],
                "must_avoid": avoid_actions,
                "expected_effect": "Conservative reward adjustment guided by diagnostics.",
                "risk_analysis": "Fallback reflection is heuristic and should be checked by validator and outcome rollback.",
            },
            "auditor_hints": {
                "package_should_be_rejected_if": [],
                "package_should_be_accepted_if": [],
                "safety_notes": [],
            },
        }
