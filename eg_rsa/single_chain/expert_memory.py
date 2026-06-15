from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .json_tools import write_json


def build_expert_memory_context(
    current_evidence: Dict[str, Any],
    best_evidence: Dict[str, Any] | None = None,
    retrieved_memory: List[Dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    behavior = current_evidence.get("behavior_summary", {}) or {}
    metrics = current_evidence.get("primary_metrics", {}) or {}
    comp = current_evidence.get("component_summary", {}) or {}
    action = behavior.get("dominant_action", {}) or {}
    action_space_report = behavior.get("action_space_report", {}) or {}
    action_id = str(action.get("action"))
    action_prob = _num(action.get("probability"))
    success_rate = _num(behavior.get("success_like_terminal_rate", metrics.get("success_like_terminal_rate")))
    episode_len = _num(metrics.get("episode_length"))
    components = comp.get("component_means", {}) or {}

    hard_constraints: List[str] = []
    diagnosis_questions: List[str] = []
    failed_patterns: List[Dict[str, Any]] = []

    mostly_negative = _mostly_negative_components(components)
    if action_id in {"0", "0.0", "None"} and action_prob >= 0.90 and success_rate <= 0.01:
        failed_patterns.append({
            "pattern_id": "passive_dominant_action",
            "evidence": {"dominant_action": action_id, "probability": action_prob, "success_rate": success_rate},
            "interpretation": "The current reward makes minimal action competitive with trying the task.",
        })
        hard_constraints.extend([
            "Do not strengthen action, fuel, or time costs under the same dominant-action behavior.",
            "Add or strengthen a reachable positive progress signal tied to the task objective.",
            "If task achievement is detectable, keep it active rather than diagnostic-only.",
            "Explain why the next reward should change the action distribution away from the current dominant action."
        ])

    unused_actions = action_space_report.get("unused_action_keys") or []
    if unused_actions:
        failed_patterns.append({
            "pattern_id": "unused_discrete_actions",
            "evidence": {
                "unused_action_keys": unused_actions,
                "action_space_report": action_space_report,
                "success_rate": success_rate,
            },
            "interpretation": "Some available actions were never selected during evaluation; a necessary control mode may be unattractive under the reward.",
        })
        hard_constraints.extend([
            "Explicitly analyze every unused discrete action and whether any is necessary for task success.",
            "Do not only reduce penalties; make the reward create a reason to try currently unused necessary actions.",
            "The next expected_behavior must predict which unused action usage should increase and why."
        ])

    if episode_len > 0 and episode_len < 150 and success_rate <= 0.01 and mostly_negative:
        failed_patterns.append({
            "pattern_id": "short_episode_dense_penalty",
            "evidence": {"episode_length": episode_len, "success_rate": success_rate},
            "interpretation": "Longer attempts may accumulate more penalty than early termination.",
        })
        hard_constraints.extend([
            "Do not repeat a reward made mainly of absolute dense penalties.",
            "Prefer progress-delta or milestone-style feedback over only negative state costs.",
            "Make useful longer attempts able to score better than short bad episodes."
        ])

    dominant = comp.get("dominant_components_by_abs_return", []) or []
    if dominant:
        top = dominant[0]
        top_abs = _num(top.get("abs_return"))
        second_abs = _num(dominant[1].get("abs_return")) if len(dominant) > 1 else 0.0
        if top_abs > 0 and (second_abs == 0.0 or top_abs >= 3.0 * max(second_abs, 1e-9)):
            failed_patterns.append({
                "pattern_id": "single_component_dominance",
                "evidence": {"top_component": top},
                "interpretation": "One component may dominate the training signal.",
            })
            hard_constraints.append("Estimate component episode-scale budgets and reduce any auxiliary term that dominates the objective signal.")

    diagnosis_questions.extend([
        "What behavior is the policy optimizing under the current reward?",
        "What behavior required for the task is being avoided?",
        "Which component makes the avoided behavior costly?",
        "Which positive signal is missing, too sparse, or too weak?",
        "Are any available actions unused, and could an unused action be required for success?",
        "Which previous edit pattern must not be repeated?"
    ])

    retrieved_memory = retrieved_memory or []
    for item in retrieved_memory:
        acc = item.get("acceptance", {}) or {}
        if acc.get("rejected"):
            lesson = item.get("lesson")
            if lesson:
                hard_constraints.append("Avoid repeating rejected edit pattern: " + str(lesson)[:400])

    context = {
        "file_type": "expert_memory_context",
        "diagnosis_questions": diagnosis_questions,
        "failed_patterns": failed_patterns,
        "hard_constraints_for_next_revision": _dedupe(hard_constraints),
        "component_balance_report": _component_balance_report(components, dominant),
        "action_space_report": action_space_report,
        "best_reference": _best_reference(best_evidence),
    }
    if output_path is not None:
        write_json(output_path, context)
    return context


def _component_balance_report(components: Dict[str, Any], dominant: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = []
    for k, v in components.items():
        try:
            vals.append((k, float(v)))
        except Exception:
            continue
    pos = [x for x in vals if x[1] > 0]
    neg = [x for x in vals if x[1] < 0]
    return {
        "num_positive_components": len(pos),
        "num_negative_components": len(neg),
        "mostly_negative": len(neg) > 0 and len(pos) == 0,
        "dominant_components_by_abs_return": dominant[:5],
    }


def _best_reference(best: Dict[str, Any] | None) -> Dict[str, Any]:
    if not best:
        return {}
    return {
        "candidate_id": best.get("candidate_id"),
        "primary_metrics": best.get("primary_metrics", {}),
        "behavior_summary": best.get("behavior_summary", {}),
    }


def _mostly_negative_components(components: Dict[str, Any]) -> bool:
    vals = []
    for v in components.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return False
    positives = [v for v in vals if v > 1e-9]
    negatives = [v for v in vals if v < -1e-9]
    return len(negatives) > 0 and len(positives) == 0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _dedupe(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
