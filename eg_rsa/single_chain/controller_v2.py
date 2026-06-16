from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .controller import SearchController as BaseSearchController
from .expert_action_planner import build_expert_action_plan


class SearchController(BaseSearchController):
    """Search controller with audit-aware search mode and acceptance checks."""

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
            decision.setdefault("elite_blocked_reasons", []).extend(gate.get("blocking_reasons", []))
        if not gate.get("parent_allowed", True) and decision.get("accepted_as_parent"):
            decision["accepted_as_parent"] = False
            decision["rejected"] = True
            decision.setdefault("rejection_reasons", []).extend(gate.get("blocking_reasons", []))
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
            "reason": "Search mode is produced by deterministic expert_audit_pack and follows the selected next parent.",
        }
        return decision


def _search_mode(evidence: Dict[str, Any]) -> Dict[str, Any]:
    audit = evidence.get("expert_audit_pack", {}) or {}
    return audit.get("search_mode_recommendation", {}) or {}


def _acceptance_gate(candidate: Dict[str, Any], best: Dict[str, Any]) -> Dict[str, Any]:
    primary = candidate.get("primary_metrics", {}) or {}
    comp = (candidate.get("component_summary", {}) or {}).get("component_means", {}) or {}
    audit = candidate.get("expert_audit_pack", {}) or {}
    plan = build_expert_action_plan(candidate, audit)

    fitness = _num(primary.get("fitness_score"))
    generated = _num(primary.get("generated_reward"))
    best_fitness = _num((best.get("primary_metrics", {}) or {}).get("fitness_score"))
    objective_bonus = _num(comp.get("objective_bonus"))
    success_flag = _num(comp.get("success_flag"))
    gap = generated - fitness

    alignment_risks = set((audit.get("objective_signal_alignment_audit", {}) or {}).get("risks", []) or [])
    payment_risks = set((audit.get("reward_payment_audit", {}) or {}).get("risks", []) or [])
    scale_risks = set((audit.get("reward_scale_audit", {}) or {}).get("risks", []) or [])

    blocking_reasons: List[str] = []
    warnings: List[str] = []
    severe_proxy_mismatch = bool(
        objective_bonus > 120.0
        or success_flag > 1.2
        or (generated > 0 and gap > max(500.0, 3.0 * max(abs(fitness), 1.0)))
        or "reward_success_signal_misaligned_with_evaluator" in alignment_risks
    )
    if severe_proxy_mismatch:
        blocking_reasons.append("reward elite blocked: candidate has high proxy-payment or objective-alignment risk")
    if "single_component_dominance" in scale_risks and objective_bonus > 120.0:
        blocking_reasons.append("reward elite blocked: objective_bonus dominates beyond one terminal-event scale")
    if "repeatable_positive_without_success" in payment_risks and objective_bonus > 120.0:
        blocking_reasons.append("reward elite blocked: repeatable positive payments exist without reliable terminal success")

    model_selection = candidate.get("model_selection", {}) or {}
    if model_selection.get("used_best_checkpoint"):
        warnings.append("candidate evidence uses best evaluation checkpoint rather than final checkpoint")
    raw_final = candidate.get("raw_final_eval_summary", {}) or {}
    if raw_final and _num(raw_final.get("fitness_score")) + 50.0 < fitness:
        warnings.append("final checkpoint was substantially worse than selected checkpoint")

    reward_elite_allowed = not blocking_reasons and bool(plan.get("reward_elite_allowed", True))
    policy_reference_allowed = bool(plan.get("policy_reference_allowed") or (severe_proxy_mismatch and fitness > max(0.0, best_fitness - 50.0)))
    parent_allowed = True
    if severe_proxy_mismatch and fitness < best_fitness - 50.0:
        parent_allowed = False
        blocking_reasons.append("parent blocked: risky candidate also regressed far below elite fitness")

    return {
        "file_type": "expert_acceptance_gate",
        "reward_elite_allowed": reward_elite_allowed,
        "parent_allowed": parent_allowed,
        "policy_reference_allowed": policy_reference_allowed,
        "blocking_reasons": _dedupe(blocking_reasons),
        "warnings": _dedupe(warnings),
        "risk_metrics": {
            "fitness_score": fitness,
            "best_fitness_score": best_fitness,
            "generated_reward": generated,
            "generated_minus_fitness_gap": gap,
            "objective_bonus_mean": objective_bonus,
            "success_flag_mean": success_flag,
            "alignment_risks": sorted(alignment_risks),
            "payment_risks": sorted(payment_risks),
            "scale_risks": sorted(scale_risks),
        },
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
