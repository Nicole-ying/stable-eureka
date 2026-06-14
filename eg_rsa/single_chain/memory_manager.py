from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .json_tools import write_json


class MemoryManager:
    """Simple JSONL memory for single-chain reward search.

    The manager records measured transitions. It does not invent metrics. LLMs may
    consume retrieved memories, but metric deltas and acceptance labels are
    computed deterministically here.
    """

    def __init__(self, memory_path: str | Path, top_k: int = 5):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_k = int(top_k)

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in self.memory_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def retrieve(self, evidence: Dict[str, Any], reflection: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        records = self.load_all()
        if not records:
            return []
        query_labels = _query_labels(evidence, reflection or {})
        scored = []
        for record in records:
            score = _memory_score(record, query_labels)
            scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = [_compress_memory(record) for score, record in scored if score > 0][: self.top_k]
        if not retrieved:
            retrieved = [_compress_memory(record) for _, record in scored[: self.top_k]]
        return retrieved

    def append_transition(
        self,
        iteration_from: int,
        iteration_to: int,
        before_evidence: Dict[str, Any],
        after_evidence: Dict[str, Any],
        reflection: Dict[str, Any],
        revision: Dict[str, Any],
        output_copy_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        before_metrics = before_evidence.get("primary_metrics", {})
        after_metrics = after_evidence.get("primary_metrics", {})
        delta = _metric_delta(before_metrics, after_metrics)
        accepted = delta.get("fitness_score", 0.0) > 0.0
        record = {
            "memory_type": "reward_transition",
            "iteration_from": iteration_from,
            "iteration_to": iteration_to,
            "parent_candidate_id": before_evidence.get("candidate_id"),
            "candidate_id": after_evidence.get("candidate_id"),
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "metric_delta": delta,
            "before_behavior": before_evidence.get("behavior_summary", {}),
            "after_behavior": after_evidence.get("behavior_summary", {}),
            "reflection_diagnosis": reflection.get("diagnosis", {}),
            "edit_summary": revision.get("edit_summary", {}),
            "acceptance": {
                "accepted_as_elite": accepted,
                "reason": "fitness_score improved" if accepted else "fitness_score did not improve",
            },
            "reuse_policy": "positive_reference" if accepted else "negative_constraint",
            "lesson": _build_rule_lesson(before_evidence, after_evidence, reflection, revision, delta, accepted),
        }
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if output_copy_path is not None:
            write_json(output_copy_path, record)
        return record


def _query_labels(evidence: Dict[str, Any], reflection: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    behavior = evidence.get("behavior_summary", {})
    diagnosis = reflection.get("diagnosis", {})
    for value in [
        behavior.get("dominant_behavior"),
        diagnosis.get("outcome_label"),
        diagnosis.get("main_problem"),
    ]:
        if isinstance(value, str) and value:
            labels.append(value.lower())
    for hint in evidence.get("automatic_diagnosis_hints", []):
        if isinstance(hint, str):
            labels.append(hint.lower())
    return labels


def _memory_score(record: Dict[str, Any], labels: List[str]) -> float:
    text = json.dumps(record, ensure_ascii=False).lower()
    score = 0.0
    for label in labels:
        for token in label.replace("_", " ").split():
            if len(token) >= 5 and token in text:
                score += 1.0
    if record.get("reuse_policy") == "positive_reference":
        score += 0.2
    return score


def _compress_memory(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "memory_type": record.get("memory_type"),
        "iteration_from": record.get("iteration_from"),
        "iteration_to": record.get("iteration_to"),
        "metric_delta": record.get("metric_delta"),
        "before_behavior": record.get("before_behavior"),
        "after_behavior": record.get("after_behavior"),
        "edit_summary": record.get("edit_summary"),
        "acceptance": record.get("acceptance"),
        "reuse_policy": record.get("reuse_policy"),
        "lesson": record.get("lesson"),
    }


def _metric_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    delta: Dict[str, float] = {}
    for key in keys:
        try:
            delta[key] = float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        except Exception:
            continue
    return delta


def _build_rule_lesson(
    before_evidence: Dict[str, Any],
    after_evidence: Dict[str, Any],
    reflection: Dict[str, Any],
    revision: Dict[str, Any],
    delta: Dict[str, float],
    accepted: bool,
) -> str:
    before_behavior = before_evidence.get("behavior_summary", {}).get("dominant_behavior", "unknown")
    after_behavior = after_evidence.get("behavior_summary", {}).get("dominant_behavior", "unknown")
    edit_scope = revision.get("edit_summary", {}).get("edit_scope", "unknown")
    changed = revision.get("edit_summary", {}).get("changed_design") or revision.get("edit_summary", {}).get("changed_active_terms") or []
    verdict = "improved" if accepted else "did not improve"
    return (
        f"A {edit_scope} reward revision from behavior '{before_behavior}' to '{after_behavior}' {verdict} "
        f"fitness_score by {delta.get('fitness_score', 0.0):.3f}. Changed design summary: {changed}. "
        f"Original diagnosis: {reflection.get('diagnosis', {}).get('main_problem', 'unknown')}."
    )
