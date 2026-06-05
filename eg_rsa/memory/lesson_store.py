from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class LessonStore:
    """JSONL store for distilled reward-editing lessons.

    Lesson cards are compact reusable experience distilled from raw memory cards.
    They are intentionally separate from raw memory so old experiment records
    stay compatible and inspectable.

    A lesson is not automatically a recommendation.  Retrieval ranks lessons by
    outcome evidence and applicability, while keeping weak/failed lessons visible
    as cautionary evidence.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        lessons: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lessons.append(json.loads(line))
        return lessons

    def append(self, lesson: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(lesson, ensure_ascii=False) + "\n")

    def retrieve(
        self,
        failure_modes: Iterable[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_modes = set(failure_modes)
        scored = []
        for lesson in self.load():
            applicability = lesson.get("applicability", {})
            modes = set(applicability.get("failure_modes", []))
            overlap = len(query_modes & modes)
            quality = lesson.get("quality", {})
            reuse_confidence = float(quality.get("reuse_confidence", 0.0) or 0.0)
            evidence_strength = float(quality.get("evidence_strength", 0.0) or 0.0)
            lesson_quality = quality.get("lesson_quality", "uncertain")

            score = overlap * 10.0 + evidence_strength
            if lesson_quality == "strong_positive":
                score += 5.0 * reuse_confidence
            elif lesson_quality == "moderate_positive":
                score += 2.0 * reuse_confidence
            elif lesson_quality in {"weak_positive", "uncertain"}:
                score += 0.5 * reuse_confidence
            elif lesson_quality in {"failed", "harmful"}:
                # Failed lessons should remain retrievable as warnings, but they
                # must not outrank strong positive lessons as recommendations.
                score += 1.0

            if score > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]


def build_lesson_from_memory_card(card: Dict[str, Any]) -> Dict[str, Any]:
    metadata = card.get("metadata", {})
    agent_analysis = metadata.get("agent_analysis", {})
    distilled = agent_analysis.get("distilled_lessons", {})
    outcome = card.get("outcome", {})
    delta = outcome.get("delta", {})
    edit_plan = card.get("edit_plan", [])
    quality = _lesson_quality(delta=delta, edit_plan=edit_plan)

    return {
        "lesson_id": f"lesson_{card.get('memory_id', 'unknown')}",
        "source_memory_id": card.get("memory_id"),
        "lesson_type": quality["lesson_type"],
        "quality": quality,
        "applicability": {
            "failure_modes": card.get("failure_modes", []),
            "dominant_component": card.get("reward_attribution", {}).get("dominant_component"),
            "dominant_component_ratio": card.get("reward_attribution", {}).get("dominant_component_ratio"),
        },
        "edit_plan": edit_plan,
        "evidence": {
            "outcome_status": outcome.get("status"),
            "before": outcome.get("before", {}),
            "after": outcome.get("after", {}),
            "outcome_delta": delta,
        },
        "agent_lessons": distilled,
        "recommendation": {
            "what_worked": distilled.get("what_worked", []),
            "what_failed": distilled.get("what_failed", []),
            "avoid_next": distilled.get("avoid_next", []),
            "recommend_next": distilled.get("recommend_next", []),
            "applicability_notes": distilled.get("applicability_notes", []),
        },
        "summary": card.get("lesson", ""),
    }


def _lesson_quality(delta: Dict[str, Any], edit_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evidence-gated lesson quality.

    This deliberately avoids trusting the LLM's confidence as proof that a lesson
    is reusable.  Reuse confidence is derived from measured task-proxy and
    hack-risk deltas only.  Thresholds are generic and can later move to config.
    """

    task_delta = float(delta.get("task_score", 0.0) or 0.0)
    hack_delta = float(delta.get("hack_score", 0.0) or 0.0)
    magnitude = abs(task_delta) + abs(hack_delta)

    if not edit_plan:
        lesson_quality = "decision_record"
        lesson_type = "no_edit_record"
        reuse_confidence = 0.0
    elif task_delta >= 0.20 or (task_delta >= 0.10 and hack_delta < 0):
        lesson_quality = "strong_positive"
        lesson_type = "effective_edit"
        reuse_confidence = 0.85
    elif task_delta >= 0.05 and hack_delta <= 0:
        lesson_quality = "moderate_positive"
        lesson_type = "effective_edit"
        reuse_confidence = 0.55
    elif task_delta > 0.0 and hack_delta <= 0:
        lesson_quality = "weak_positive"
        lesson_type = "weak_edit"
        reuse_confidence = 0.25
    elif task_delta <= 0.0 and hack_delta >= 0.0:
        lesson_quality = "failed"
        lesson_type = "failed_or_weak_edit"
        reuse_confidence = 0.0
    elif hack_delta > 0.0:
        lesson_quality = "harmful"
        lesson_type = "harmful_edit"
        reuse_confidence = 0.0
    else:
        lesson_quality = "uncertain"
        lesson_type = "mixed_outcome_edit"
        reuse_confidence = 0.15

    evidence_strength = min(1.0, 0.35 + magnitude)
    if lesson_quality in {"failed", "harmful"}:
        # A failed result can be strong evidence for avoidance even though it has
        # zero reuse confidence as a recommendation.
        evidence_strength = min(1.0, evidence_strength + 0.25)

    return {
        "lesson_type": lesson_type,
        "lesson_quality": lesson_quality,
        "evidence_strength": float(evidence_strength),
        "reuse_confidence": float(reuse_confidence),
        "task_delta": float(task_delta),
        "hack_delta": float(hack_delta),
        "interpretation": _quality_interpretation(lesson_quality),
    }


def _quality_interpretation(lesson_quality: str) -> str:
    mapping = {
        "strong_positive": "Reusable only when applicability matches; supported by clear proxy improvement.",
        "moderate_positive": "Potentially reusable, but should be checked against current diagnostics.",
        "weak_positive": "Weak evidence; treat as hypothesis, not a recommendation.",
        "failed": "Do not repeat as a recommendation; use as cautionary evidence.",
        "harmful": "Avoid repeating unless a later auditor provides strong counter-evidence.",
        "uncertain": "Insufficient or mixed evidence; do not treat as a stable lesson.",
        "decision_record": "Decision record without measured edit effect.",
    }
    return mapping.get(lesson_quality, "Unclassified lesson quality.")
