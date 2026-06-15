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
    audit_pack = current_evidence.get("expert_audit_pack", {}) or {}
    action = behavior.get("dominant_action", {}) or {}
    action_space_report = behavior.get("action_space_report", {}) or {}
    components = comp.get("component_means", {}) or {}

    action_id = str(action.get("action"))
    action_prob = _num(action.get("probability"))
    success_rate = _num(behavior.get("success_like_terminal_rate", metrics.get("success_like_terminal_rate")))
    episode_len = _num(metrics.get("episode_length"))

    failed_patterns: List[Dict[str, Any]] = []
    hard_constraints: List[str] = []

    unused_actions = action_space_report.get("unused_action_keys") or []
    if unused_actions:
        failed_patterns.append({
            "pattern_id": "unused_discrete_actions",
            "evidence": {"unused_action_keys": unused_actions, "action_space_report": action_space_report, "success_rate": success_rate},
            "interpretation": "Some discrete controls were not selected during evaluation.",
        })
        hard_constraints.extend([
            "Analyze every unused discrete control and whether it is required for success.",
            "Create a reachable reason to try unused required controls.",
            "The next expected_behavior must state which unused control should increase and why.",
        ])

    if action_id in {"0", "0.0", "None"} and action_prob >= 0.90 and success_rate <= 0.01:
        failed_patterns.append({
            "pattern_id": "passive_dominant_action",
            "evidence": {"dominant_action": action_id, "probability": action_prob, "success_rate": success_rate},
            "interpretation": "Minimal-control behavior is currently competitive.",
        })
        hard_constraints.append("Add reachable progress feedback tied to the primary objective.")

    if episode_len > 0 and episode_len < 150 and success_rate <= 0.01 and _mostly_negative_components(components):
        failed_patterns.append({
            "pattern_id": "short_episode_dense_penalty",
            "evidence": {"episode_length": episode_len, "success_rate": success_rate},
            "interpretation": "Useful longer attempts may be under-valued.",
        })
        hard_constraints.append("Prefer progress-delta or milestone feedback over only absolute state costs.")

    dominant = comp.get("dominant_components_by_abs_return", []) or []
    if dominant:
        top = dominant[0]
        top_abs = _num(top.get("abs_return"))
        second_abs = _num(dominant[1].get("abs_return")) if len(dominant) > 1 else 0.0
        if top_abs > 0 and (second_abs == 0.0 or top_abs >= 3.0 * max(second_abs, 1e-9)):
            failed_patterns.append({
                "pattern_id": "single_component_dominance",
                "evidence": {"top_component": top},
                "interpretation": "One component may dominate the episode-scale signal.",
            })
            hard_constraints.append("Estimate episode-scale budgets and reduce auxiliary dominance.")

    for report_name in ["reward_scale_audit", "reward_payment_audit", "objective_signal_alignment_audit"]:
        report = audit_pack.get(report_name, {}) or {}
        if report.get("risks"):
            failed_patterns.append({
                "pattern_id": report_name + "_risk",
                "evidence": report,
                "interpretation": "Deterministic audit found a risk that should guide the next edit.",
            })

    phase_report = audit_pack.get("trajectory_phase_report", {}) or {}
    failure_phase = phase_report.get("failure_phase")
    if failure_phase and failure_phase not in {"unknown", "success_discovered"}:
        failed_patterns.append({
            "pattern_id": "trajectory_phase_bottleneck",
            "evidence": phase_report,
            "interpretation": "A trajectory phase bottleneck is visible in evaluation evidence.",
        })

    hard_constraints.extend(audit_pack.get("hard_constraints", []) or [])

    retrieved_memory = retrieved_memory or []
    for item in retrieved_memory:
        acc = item.get("acceptance", {}) or {}
        if acc.get("accepted_as_elite") is False and item.get("lesson"):
            hard_constraints.append("Avoid prior rejected pattern: " + str(item.get("lesson"))[:400])

    context = {
        "file_type": "expert_memory_context",
        "diagnosis_questions": [
            "What behavior is the policy optimizing under the current reward?",
            "What required behavior is not yet expressed?",
            "Which component is shaping that behavior most strongly?",
            "Which positive signal is missing, sparse, or weak?",
            "Are any available discrete controls unused?",
            "If all controls are used, which trajectory phase remains unsolved?",
            "Do reward-defined success signals align with evaluator success?",
            "Which previous edit pattern should be avoided?",
        ],
        "failed_patterns": failed_patterns,
        "hard_constraints_for_next_revision": _dedupe(hard_constraints),
        "component_balance_report": _component_balance_report(components, dominant),
        "action_space_report": action_space_report,
        "expert_audit_pack": audit_pack,
        "search_mode_recommendation": audit_pack.get("search_mode_recommendation", {}) if audit_pack else {},
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
        "expert_audit_pack": best.get("expert_audit_pack", {}),
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
        if not item:
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
