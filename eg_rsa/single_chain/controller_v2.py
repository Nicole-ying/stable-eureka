from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .controller import SearchController as BaseSearchController
from .expert_action_planner import build_expert_action_plan


class SearchController(BaseSearchController):
    """Search controller with final-behavior and uncertainty-aware checks."""

    def decide(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        decision = super().decide(*args, **kwargs)
        parent_evidence = kwargs.get("parent_evidence") if kwargs else None
        candidate_evidence = kwargs.get("candidate_evidence") if kwargs else None
        best_evidence = kwargs.get("best_evidence") if kwargs else None
        candidate_dir = kwargs.get("candidate_dir") if kwargs else None
        if parent_evidence is None and args:
            parent_evidence = args[0]
        if candidate_evidence is None and len(args) > 1:
            candidate_evidence = args[1]
        if best_evidence is None and len(args) > 2:
            best_evidence = args[2]
        if candidate_dir is None and len(args) > 5:
            candidate_dir = args[5]

        parent_evidence = parent_evidence or {}
        candidate_evidence = candidate_evidence or {}
        best_evidence = best_evidence or {}

        gate = _acceptance_gate(candidate_evidence, best_evidence)
        decision["expert_acceptance_gate"] = gate
        if not gate.get("reward_elite_allowed", True) and decision.get("accepted_as_elite"):
            decision["accepted_as_elite"] = False
            decision.setdefault("elite_reasons", []).extend(gate.get("reasons", []))
        if not gate.get("parent_allowed", True) and decision.get("accepted_as_parent"):
            decision["accepted_as_parent"] = False
            decision["rejected"] = True
            decision.setdefault("rejection_reasons", []).extend(gate.get("reasons", []))
            fallback_dir = decision.get("rollback", {}).get("next_parent_dir")
            if fallback_dir:
                decision["next_search_parent"]["source"] = decision.get("rollback", {}).get("next_parent_source", "elite_parent")
                decision["next_search_parent"]["dir"] = fallback_dir
        elif gate.get("policy_reference_allowed") and not gate.get("reward_elite_allowed", True):
            decision["accepted_as_policy_reference"] = True
            if candidate_dir is not None:
                decision.setdefault("policy_reference", {})["dir"] = str(Path(candidate_dir))

        parent_mode = _search_mode(parent_evidence)
        candidate_mode = _search_mode(candidate_evidence)
        decision["search_mode"] = {
            "parent_next_search_mode": parent_mode,
            "candidate_next_search_mode": candidate_mode,
            "selected_next_search_mode": candidate_mode if decision.get("accepted_as_parent") else parent_mode,
            "reason": "Search mode follows the selected next parent.",
        }
        return decision


def _search_mode(evidence: Dict[str, Any]) -> Dict[str, Any]:
    audit = evidence.get("expert_audit_pack", {}) or {}
    return audit.get("search_mode_recommendation", {}) or {}


def _acceptance_gate(candidate: Dict[str, Any], best: Dict[str, Any]) -> Dict[str, Any]:
    primary = candidate.get("primary_metrics", {}) or {}
    comp = (candidate.get("component_summary", {}) or {}).get("component_means", {}) or {}
    audit = candidate.get("expert_audit_pack", {}) or {}
    target_report = candidate.get("target_behavior_report", {}) or {}
    stability = candidate.get("checkpoint_stability_report", {}) or {}
    plan = build_expert_action_plan(candidate, audit)

    fitness = _num(primary.get("fitness_score"))
    fitness_std = _num(primary.get("fitness_score_std"))
    generated = _num(primary.get("generated_reward"))
    best_fitness = _num((best.get("primary_metrics", {}) or {}).get("fitness_score"))
    best_target = best.get("target_behavior_report", {}) or {}
    best_criteria = best_target.get("primary_behavior_criteria", {}) or {}
    best_success_rate = _num(best_criteria.get("success_like_terminal_rate", (best.get("primary_metrics", {}) or {}).get("success_like_terminal_rate")))
    best_timeout_rate = _num(best_criteria.get("timeout_rate"))
    objective_bonus = _num(comp.get("objective_bonus"))
    success_flag = _num(comp.get("success_flag"))
    gap = generated - fitness
    criteria = target_report.get("primary_behavior_criteria", {}) or {}
    success_rate = _num(criteria.get("success_like_terminal_rate", primary.get("success_like_terminal_rate")))
    timeout_rate = _num(criteria.get("timeout_rate"))
    target_success = bool(target_report.get("target_success", success_rate >= 0.5 and timeout_rate <= 0.4))
    transient_peak = bool(stability.get("transient_peak_risk"))
    uncertainty = target_report.get("auxiliary_score_stability", {}) or {}
    high_fitness_uncertainty = bool(uncertainty.get("high_fitness_uncertainty") or stability.get("final_uncertainty_risk") or stability.get("unstable_last_k_risk"))
    behavior_improves = bool(success_rate > best_success_rate + 0.1 or (best_timeout_rate > 0 and timeout_rate < best_timeout_rate - 0.1))
    score_gain = fitness - best_fitness
    noisy_small_gain = bool(high_fitness_uncertainty and score_gain > 0 and score_gain <= max(10.0, fitness_std))

    alignment_risks = set((audit.get("objective_signal_alignment_audit", {}) or {}).get("risks", []) or [])
    payment_risks = set((audit.get("reward_payment_audit", {}) or {}).get("risks", []) or [])
    scale_risks = set((audit.get("reward_scale_audit", {}) or {}).get("risks", []) or [])

    reasons: List[str] = []
    proxy_mismatch = bool(
        objective_bonus > 120.0
        or success_flag > 1.2
        or (generated > 0 and gap > max(500.0, 3.0 * max(abs(fitness), 1.0)))
        or "reward_success_signal_misaligned_with_evaluator" in alignment_risks
    )
    if proxy_mismatch:
        reasons.append("candidate reward payments are not aligned with target behavior evidence")
    if not target_success:
        reasons.append("final target behavior contract is not satisfied")
    if transient_peak:
        reasons.append("training has a transient peak that is not stable final behavior")
    if noisy_small_gain and not behavior_improves:
        reasons.append("auxiliary fitness gain is smaller than evaluator noise and behavior did not clearly improve")
    if "single_component_dominance" in scale_risks and objective_bonus > 120.0:
        reasons.append("objective bonus dominates beyond one terminal-event scale")
    if "repeatable_positive_without_success" in payment_risks and objective_bonus > 120.0:
        reasons.append("repeatable positive payments appear without reliable terminal success")
    if (candidate.get("model_selection", {}) or {}).get("used_best_checkpoint"):
        reasons.append("model selection used a peak checkpoint instead of final behavior")

    reward_elite_allowed = not reasons and bool(plan.get("reward_elite_allowed", True))
    policy_reference_allowed = bool(plan.get("policy_reference_allowed") or (success_rate > 0.0 and fitness > max(0.0, best_fitness - 50.0)))
    parent_allowed = True
    if timeout_rate >= 0.8 and success_rate <= 0.01 and fitness < best_fitness:
        parent_allowed = False
        reasons.append("final behavior is timeout-dominated and does not improve auxiliary fitness")
    if proxy_mismatch and fitness < best_fitness - 50.0:
        parent_allowed = False
        reasons.append("candidate regressed far below elite auxiliary fitness")

    return {
        "file_type": "expert_acceptance_gate",
        "reward_elite_allowed": reward_elite_allowed,
        "parent_allowed": parent_allowed,
        "policy_reference_allowed": policy_reference_allowed,
        "reasons": _dedupe(reasons),
        "risk_metrics": {
            "fitness_score_auxiliary": fitness,
            "fitness_score_std": fitness_std,
            "best_fitness_score_auxiliary": best_fitness,
            "score_gain": score_gain,
            "high_fitness_uncertainty": high_fitness_uncertainty,
            "noisy_small_gain": noisy_small_gain,
            "behavior_improves": behavior_improves,
            "generated_reward_diagnostic": generated,
            "generated_minus_fitness_gap_diagnostic": gap,
            "objective_bonus_mean": objective_bonus,
            "success_flag_mean": success_flag,
            "final_success_like_terminal_rate": success_rate,
            "best_success_like_terminal_rate": best_success_rate,
            "timeout_rate": timeout_rate,
            "best_timeout_rate": best_timeout_rate,
            "target_success": target_success,
            "transient_peak_risk": transient_peak,
            "alignment_risks": sorted(alignment_risks),
            "payment_risks": sorted(payment_risks),
            "scale_risks": sorted(scale_risks),
        },
        "target_behavior_report": target_report,
        "checkpoint_stability_report": stability,
        "expert_action_plan": plan,
    }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
