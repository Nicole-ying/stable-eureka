from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class LessonStore:
    """JSONL store for distilled reward-editing lessons.

    Lesson cards are compact reusable experience distilled from raw memory cards.
    They are intentionally separate from raw memory so old experiment records
    stay compatible and inspectable.
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
            confidence = float(lesson.get("confidence", 0.0) or 0.0)
            outcome = lesson.get("evidence", {}).get("outcome_delta", {})
            task_delta = float(outcome.get("task_score", 0.0) or 0.0)
            hack_delta = float(outcome.get("hack_score", 0.0) or 0.0)
            score = overlap * 10.0 + confidence
            if task_delta > 0:
                score += min(3.0, task_delta * 2.0)
            if hack_delta < 0:
                score += min(3.0, abs(hack_delta) * 4.0)
            if score > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]


def build_lesson_from_memory_card(card: Dict[str, Any]) -> Dict[str, Any]:
    metadata = card.get("metadata", {})
    agent_analysis = metadata.get("agent_analysis", {})
    distilled = agent_analysis.get("distilled_lessons", {})
    editor = agent_analysis.get("reward_editor", {})
    diagnostic = agent_analysis.get("diagnostic_analysis", {})
    outcome = card.get("outcome", {})
    delta = outcome.get("delta", {})
    edit_plan = card.get("edit_plan", [])

    lesson_type = _lesson_type(delta, edit_plan)
    confidence = _lesson_confidence(delta, diagnostic, editor)
    return {
        "lesson_id": f"lesson_{card.get('memory_id', 'unknown')}",
        "source_memory_id": card.get("memory_id"),
        "lesson_type": lesson_type,
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
        "confidence": confidence,
        "summary": card.get("lesson", ""),
    }


def _lesson_type(delta: Dict[str, Any], edit_plan: List[Dict[str, Any]]) -> str:
    if not edit_plan:
        return "no_edit_record"
    task_delta = float(delta.get("task_score", 0.0) or 0.0)
    hack_delta = float(delta.get("hack_score", 0.0) or 0.0)
    if task_delta > 0 and hack_delta <= 0:
        return "effective_edit"
    if task_delta <= 0 and hack_delta >= 0:
        return "failed_or_weak_edit"
    return "mixed_outcome_edit"


def _lesson_confidence(delta: Dict[str, Any], diagnostic: Dict[str, Any], editor: Dict[str, Any]) -> float:
    base = 0.5
    try:
        base = max(base, float(diagnostic.get("confidence", 0.0) or 0.0))
    except (TypeError, ValueError):
        pass
    task_delta = abs(float(delta.get("task_score", 0.0) or 0.0))
    hack_delta = abs(float(delta.get("hack_score", 0.0) or 0.0))
    if editor.get("edit_decision") == "edit":
        base += 0.05
    base += min(0.2, task_delta * 0.2 + hack_delta * 0.3)
    return float(max(0.0, min(1.0, base)))
