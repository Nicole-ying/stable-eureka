from __future__ import annotations

from typing import Any, Dict, List

from eg_rsa.tools.schema_diff import diff_schemas


class OutcomeLessonBuilder:
    """Build reusable lessons from measured transitions.

    This module turns a measured transition into structured evidence that can be
    retrieved by future agent decisions.
    """

    name = "outcome_lesson_builder"

    @staticmethod
    def build(
        before_schema: Any,
        after_schema: Any,
        edit_plan: List[Dict[str, Any]],
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
        outcome_decision: Dict[str, Any] | None = None,
        attribution_after: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        outcome_decision = outcome_decision or {}
        attribution_after = attribution_after or {}
        delta = OutcomeLessonBuilder._delta(before_metrics, after_metrics)
        schema_diff = diff_schemas(before_schema, after_schema)
        lesson_type = OutcomeLessonBuilder._lesson_type(delta, outcome_decision, attribution_after)

        return {
            "tool": OutcomeLessonBuilder.name,
            "lesson_type": lesson_type,
            "quality": OutcomeLessonBuilder._quality(lesson_type, delta),
            "edit_plan": edit_plan or [],
            "schema_diff": schema_diff,
            "metrics_before": before_metrics,
            "metrics_after": after_metrics,
            "metric_delta": delta,
            "outcome_decision": outcome_decision,
            "mechanism_summary": OutcomeLessonBuilder._mechanism_summary(lesson_type, delta, attribution_after),
            "future_guidance": OutcomeLessonBuilder._future_guidance(lesson_type, attribution_after),
            "applicability": {
                "dominant_component": attribution_after.get("dominant_component"),
                "failure_modes": after_metrics.get("failure_modes", []),
                "true_hack_risk": after_metrics.get("true_hack_risk", False),
                "terminal_goal_evidence": after_metrics.get("terminal_goal_evidence", False),
            },
        }

    @staticmethod
    def _delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
        keys = [
            "task_score",
            "semantic_score",
            "selection_score",
            "hack_score",
            "success_episode_rate",
            "terminal_reward_paid_episode_rate",
            "stable_landing_episode_rate",
            "safe_contact_episode_rate",
        ]
        out: Dict[str, float] = {}
        for key in keys:
            out[key] = float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)
        return out

    @staticmethod
    def _lesson_type(delta: Dict[str, float], outcome: Dict[str, Any], attribution: Dict[str, Any]) -> str:
        if bool(outcome.get("rollback_recommended", False)):
            dominant_ratio = float(attribution.get("dominant_component_ratio", 0.0) or 0.0)
            if dominant_ratio >= 0.7:
                return "dominance_regression_lesson"
            return "regression_lesson"
        if delta.get("selection_score", 0.0) > 0.1 or delta.get("semantic_score", 0.0) > 0.1:
            return "effective_edit_lesson"
        if max(delta.get("task_score", 0.0), delta.get("semantic_score", 0.0)) > 0.0:
            return "small_positive_lesson"
        return "uncertain_lesson"

    @staticmethod
    def _quality(lesson_type: str, delta: Dict[str, float]) -> Dict[str, Any]:
        if lesson_type == "effective_edit_lesson":
            return {"reuse_confidence": 0.8, "evidence_strength": 0.8}
        if lesson_type == "small_positive_lesson":
            return {"reuse_confidence": 0.4, "evidence_strength": 0.5}
        if lesson_type in {"regression_lesson", "dominance_regression_lesson"}:
            return {"reuse_confidence": 0.75, "evidence_strength": 0.85}
        return {"reuse_confidence": 0.2, "evidence_strength": 0.3}

    @staticmethod
    def _mechanism_summary(lesson_type: str, delta: Dict[str, float], attribution: Dict[str, Any]) -> str:
        dominant = attribution.get("dominant_component")
        ratio = attribution.get("dominant_component_ratio")
        if lesson_type == "dominance_regression_lesson":
            return f"Outcome regressed while component {dominant} dominated reward attribution with ratio={ratio}."
        if lesson_type == "regression_lesson":
            return "Outcome regressed after edit; inspect schema diff and avoid repeating this transition without new evidence."
        if lesson_type == "effective_edit_lesson":
            return "Edit improved internal task or semantic evidence and can be reused when applicability matches."
        if lesson_type == "small_positive_lesson":
            return "Edit produced small positive evidence; continue candidate rather than immediate rollback when no true hack risk exists."
        return "Outcome was ambiguous; more evidence or multi-seed evaluation is needed."

    @staticmethod
    def _future_guidance(lesson_type: str, attribution: Dict[str, Any]) -> List[str]:
        if lesson_type == "dominance_regression_lesson":
            return [
                "Run scale audit before adding or increasing dense components.",
                "Start new penalties with small weights and candidate sweep.",
                "Do not let a new dense term dominate terminal incentives.",
            ]
        if lesson_type == "regression_lesson":
            return [
                "Retrieve this lesson before similar edit operators.",
                "Prefer conservative local refinement or continue_training.",
            ]
        if lesson_type == "effective_edit_lesson":
            return [
                "Reuse when failure modes and component attribution match.",
                "Keep atomic edit package coherence.",
            ]
        return ["Gather more evidence before strong policy update."]
