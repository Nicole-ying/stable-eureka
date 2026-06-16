from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .json_tools import write_json


def build_expert_action_plan(
    current_evidence: Dict[str, Any],
    audit_pack: Dict[str, Any] | None = None,
    retrieved_memory: List[Dict[str, Any]] | None = None,
    consecutive_rejections: int = 0,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Translate expert diagnosis into an executable edit-and-validation contract.

    The planner is behavior-contract driven: fitness_score is useful auxiliary
    evidence, but reward quality is judged by whether final policy behavior truly
    reaches the task objective without proxy farming or transient checkpoint peaks.
    """
    audit_pack = audit_pack or current_evidence.get("expert_audit_pack", {}) or {}
    retrieved_memory = retrieved_memory or []

    primary = current_evidence.get("primary_metrics", {}) or {}
    behavior = current_evidence.get("behavior_summary", {}) or {}
    components = (current_evidence.get("component_summary", {}) or {}).get("component_means", {}) or {}
    target_report = current_evidence.get("target_behavior_report", {}) or {}
    stability_report = current_evidence.get("checkpoint_stability_report", {}) or {}
    trajectory = audit_pack.get("trajectory_phase_report", {}) or {}
    payment = audit_pack.get("reward_payment_audit", {}) or {}
    scale = audit_pack.get("reward_scale_audit", {}) or {}
    alignment = audit_pack.get("objective_signal_alignment_audit", {}) or {}
    search_mode = audit_pack.get("search_mode_recommendation", {}) or {}

    fitness = _num(primary.get("fitness_score"))
    generated = _num(primary.get("generated_reward"))
    success_rate = _num(primary.get("success_like_terminal_rate"))
    timeout_rate = _num((target_report.get("primary_behavior_criteria", {}) or {}).get("timeout_rate"))
    episode_length = _num(primary.get("episode_length"))
    objective_bonus = _num(components.get("objective_bonus"))
    success_flag = _num(components.get("success_flag"))
    gap = generated - fitness
    coverage = _num((behavior.get("action_space_report", {}) or {}).get("coverage_ratio"))
    stable_touchdown_rate = _num(trajectory.get("stable_touchdown_rate"))
    transient_peak_risk = bool(stability_report.get("transient_peak_risk"))
    target_success = bool(target_report.get("target_success", success_rate >= 0.5 and timeout_rate <= 0.4))

    risks = []
    if scale.get("risks"):
        risks.extend(["scale:" + str(x) for x in scale.get("risks", [])])
    if payment.get("risks"):
        risks.extend(["payment:" + str(x) for x in payment.get("risks", [])])
    if alignment.get("risks"):
        risks.extend(["alignment:" + str(x) for x in alignment.get("risks", [])])
    if transient_peak_risk:
        risks.append("checkpoint:transient_peak_risk")
    if not target_success:
        risks.append("behavior:target_behavior_gap")

    reward_payment_risk = bool(
        objective_bonus > 120.0
        or success_flag > 1.2
        or (generated > 0 and gap > max(500.0, 3.0 * max(abs(fitness), 1.0)))
        or "alignment:reward_success_signal_misaligned_with_evaluator" in risks
    )
    behavior_contract_risk = bool(
        not target_success
        or transient_peak_risk
        or (timeout_rate > 0.4 and success_rate < 0.5)
    )
    instrumentation_gap_risk = bool(
        success_rate <= 0.01
        and stable_touchdown_rate >= 0.5
        and fitness > 0
    )
    consecutive_failure_risk = consecutive_rejections >= 5

    action_type = "balanced_reward_revision"
    edit_scope = "medium"
    required_edits: List[str] = []
    forbidden_edits: List[str] = []
    acceptance_gates: List[Dict[str, Any]] = []
    validation_tests: List[str] = []
    rationale: List[str] = []
    reward_elite_allowed = True
    policy_reference_allowed = False

    if behavior_contract_risk:
        action_type = "target_behavior_repair"
        edit_scope = "medium"
        reward_elite_allowed = False
        policy_reference_allowed = bool(success_rate > 0.0 or fitness > 0.0)
        required_edits.extend(target_report.get("required_reward_fixes", []) or [])
        required_edits.extend([
            "Treat fitness_score as auxiliary evidence; optimize the reward function for stable target behavior, not for a transient score peak.",
            "Use final checkpoint behavior and target_behavior_report as the main diagnosis for the next reward edit.",
        ])
        forbidden_edits.extend([
            "Do not preserve a reward design merely because a transient checkpoint reached high fitness.",
            "Do not optimize generated_reward as a framework objective; use it only to diagnose what the policy learned to collect.",
        ])
        acceptance_gates.extend([
            {"metric": "final_success_like_terminal_rate", "operator": ">=", "threshold": 0.5, "reason": "reward quality requires stable final task success"},
            {"metric": "timeout_rate", "operator": "<=", "threshold": 0.4, "reason": "near-target timeout behavior is not task completion"},
            {"metric": "transient_peak_risk", "operator": "==", "threshold": False, "reason": "reward elite must not be based on peak-only behavior"},
        ])
        validation_tests.append("Target behavior report must show final target_success=true or an explicitly improved behavior gap.")
        rationale.append("Final behavior contract is not satisfied; reward design quality is not established by auxiliary fitness alone.")

    if reward_payment_risk:
        action_type = "repair_reward_payment" if not behavior_contract_risk else action_type
        edit_scope = "medium"
        reward_elite_allowed = False
        policy_reference_allowed = bool(policy_reference_allowed or fitness > 0)
        required_edits.extend([
            "Make completion/objective bonuses one-shot or terminal-aligned so they cannot fire every step.",
            "Remove repeatable proxy-success payment modes that can be farmed without episode completion.",
            "Keep dense progress signals bounded and smaller than episode-scale objective signals.",
            "Add diagnostics for success_flag/objective_bonus so post-training audits can verify payment count.",
        ])
        forbidden_edits.extend([
            "Do not increase raw per-step action or engine bonuses.",
            "Do not let objective_bonus exceed roughly one terminal event per episode.",
            "Do not define success only as a non-terminal state predicate if it pays a large reward repeatedly.",
        ])
        acceptance_gates.extend([
            {"metric": "success_flag_mean", "operator": "<=", "threshold": 1.2, "reason": "large success payments should not repeat many times per episode"},
            {"metric": "objective_bonus_mean", "operator": "<=", "threshold": 120.0, "reason": "one +100 objective event per episode is the expected scale"},
            {"metric": "generated_minus_fitness_gap", "operator": "<=", "threshold": 500.0, "reason": "large proxy gaps are diagnostic of possible reward-behavior mismatch"},
        ])
        validation_tests.append("Static reward validator must not detect repeatable objective_bonus or non-terminal proxy-success farming.")
        rationale.append("Reward payment or objective alignment audits found a non-trustworthy payment structure.")

    if instrumentation_gap_risk:
        required_edits.append(
            "Use terminal diagnostics from the environment wrapper when judging success; do not rely only on reward-defined success proxies."
        )
        validation_tests.append("safe_landing_terminal, crash_terminal, out_of_bounds_terminal, and timeout_terminal should be present in evaluation info.")
        rationale.append("Stable near-pad final states with success_like_terminal_rate near zero indicate an instrumentation or terminal-alignment gap.")

    if coverage >= 0.99 and success_rate <= 0.01:
        action_type = action_type if (reward_payment_risk or behavior_contract_risk) else "phase_guidance_after_action_coverage"
        required_edits.extend([
            "Focus on controlled descent, near-ground stability, and terminal touchdown rather than action coverage.",
            "Condition control incentives on reducing distance/velocity/angle in the correct phase.",
        ])
        forbidden_edits.append("Do not spend the next edit only trying to increase action coverage; coverage is already solved.")
        rationale.append("All discrete controls are used, so the bottleneck has moved to trajectory phase completion.")

    if consecutive_failure_risk:
        edit_scope = "large"
        action_type = "large_scope_restructure" if not (reward_payment_risk or behavior_contract_risk) else action_type
        required_edits.append("After repeated rejected candidates, change the reward design family rather than making another local scalar tweak.")
        validation_tests.append("The edit_summary must explain which old reward family was abandoned and why.")
        rationale.append("Several consecutive rejected candidates indicate local search stagnation.")

    for item in retrieved_memory[:5]:
        lesson = item.get("lesson") or item.get("reuse_policy") or ""
        if lesson:
            forbidden_edits.append("Avoid remembered failed pattern: " + str(lesson)[:240])

    if not required_edits:
        required_edits.extend([
            "Preserve behavior that improved final target behavior, not just peak fitness.",
            "Make one targeted reward edit with an explicit expected behavior change.",
        ])
    if not validation_tests:
        validation_tests.extend([
            "Compare final target_behavior_report before and after the edit.",
            "Use fitness_score as an auxiliary evaluator, not as the sole objective.",
        ])

    plan = {
        "file_type": "expert_action_plan",
        "action_type": action_type,
        "search_mode": search_mode.get("next_search_mode", action_type),
        "edit_scope": edit_scope,
        "reward_elite_allowed": reward_elite_allowed,
        "policy_reference_allowed": policy_reference_allowed,
        "risk_summary": {
            "reward_payment_risk": reward_payment_risk,
            "behavior_contract_risk": behavior_contract_risk,
            "instrumentation_gap_risk": instrumentation_gap_risk,
            "transient_peak_risk": transient_peak_risk,
            "consecutive_failure_risk": consecutive_failure_risk,
            "audit_risks": risks,
            "fitness_score_auxiliary": fitness,
            "generated_reward_diagnostic": generated,
            "generated_minus_fitness_gap_diagnostic": gap,
            "objective_bonus_mean": objective_bonus,
            "success_flag_mean": success_flag,
            "success_like_terminal_rate": success_rate,
            "timeout_rate": timeout_rate,
            "episode_length": episode_length,
            "stable_touchdown_rate": stable_touchdown_rate,
        },
        "target_behavior_report": target_report,
        "checkpoint_stability_report": stability_report,
        "required_edits": _dedupe(required_edits),
        "forbidden_edits": _dedupe(forbidden_edits + (search_mode.get("forbidden_edits", []) or [])),
        "required_focus": _dedupe(search_mode.get("required_focus", []) or []),
        "acceptance_gates": acceptance_gates,
        "validation_tests": _dedupe(validation_tests),
        "next_if_fail": "If final behavior still misses the target contract, use a larger reward redesign focused on terminal task completion and anti-timeout proxy farming.",
        "rationale": _dedupe(rationale),
    }
    if output_path is not None:
        write_json(output_path, plan)
    return plan


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
        text = str(item)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out
