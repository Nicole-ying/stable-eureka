from __future__ import annotations

import re
from typing import Any, Dict, List


SUCCESS_KEYS = ("success", "objective", "landing", "landed", "terminal_reward")
FAILURE_KEYS = ("crash", "unsafe", "out_of_bounds", "failure")
ACTION_HINT_KEYS = ("engine", "action", "idle", "thrust", "control")


def build_expert_audit_pack(
    evidence: Dict[str, Any],
    reward_schema: Dict[str, Any] | None = None,
    reward_code: str = "",
    task_model: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build deterministic expert audits from measured evidence.

    The audit pack is intentionally lightweight and conservative: it should not
    replace the LLM expert, but should make important expert observations explicit
    before the next reward revision is generated.
    """
    reward_schema = reward_schema or {}
    task_model = task_model or {}

    scale_audit = reward_scale_audit(evidence)
    payment_audit = reward_payment_audit(evidence, reward_schema, reward_code)
    phase_report = trajectory_phase_report(evidence, task_model)
    objective_audit = objective_signal_alignment_audit(evidence)
    search_mode = recommend_search_mode(evidence, scale_audit, payment_audit, phase_report, objective_audit)

    hard_constraints = _dedupe(
        scale_audit.get("hard_constraints", [])
        + payment_audit.get("hard_constraints", [])
        + phase_report.get("hard_constraints", [])
        + objective_audit.get("hard_constraints", [])
        + search_mode.get("hard_constraints", [])
    )
    automatic_hints = _dedupe(
        scale_audit.get("hints", [])
        + payment_audit.get("hints", [])
        + phase_report.get("hints", [])
        + objective_audit.get("hints", [])
        + search_mode.get("hints", [])
    )

    return {
        "file_type": "expert_audit_pack",
        "reward_scale_audit": scale_audit,
        "reward_payment_audit": payment_audit,
        "trajectory_phase_report": phase_report,
        "objective_signal_alignment_audit": objective_audit,
        "search_mode_recommendation": search_mode,
        "hard_constraints": hard_constraints,
        "automatic_hints": automatic_hints,
    }


def reward_scale_audit(evidence: Dict[str, Any]) -> Dict[str, Any]:
    components = evidence.get("component_summary", {}).get("component_means", {}) or {}
    dominant = evidence.get("component_summary", {}).get("dominant_components_by_abs_return", []) or []

    objective_budget = 0.0
    terminal_failure_budget = 0.0
    dense_negative_budget = 0.0
    dense_positive_budget = 0.0
    auxiliary_budget = 0.0

    for name, raw_value in components.items():
        value = _num(raw_value)
        lower = str(name).lower()
        if lower in {"fitness_score", "total_reward", "reward"}:
            continue
        if any(token in lower for token in SUCCESS_KEYS) and value > 0:
            objective_budget += value
        elif any(token in lower for token in FAILURE_KEYS) and value < 0:
            terminal_failure_budget += value
        elif value < 0:
            dense_negative_budget += value
        elif value > 0:
            dense_positive_budget += value
        auxiliary_budget += abs(value)

    top = dominant[0] if dominant else {}
    top_abs = _num(top.get("abs_return"))
    second_abs = _num(dominant[1].get("abs_return")) if len(dominant) > 1 else 0.0
    dominance_ratio = top_abs / max(second_abs, 1e-9) if top_abs > 0 else 0.0
    objective_reference = max(abs(objective_budget), 1.0)

    risks: List[str] = []
    hard_constraints: List[str] = []
    hints: List[str] = []

    if dominance_ratio >= 3.0:
        risks.append("single_component_dominance")
        hints.append(f"Reward scale audit: dominant component {top.get('name')} has abs return {top_abs:.3f}, dominance_ratio={dominance_ratio:.2f}.")
        hard_constraints.append("Do not let one auxiliary component dominate the episode-scale objective signal.")

    dense_to_objective_ratio = abs(dense_negative_budget) / objective_reference
    if dense_to_objective_ratio >= 3.0:
        risks.append("dense_penalty_dominates_objective")
        hints.append(f"Reward scale audit: dense negative budget {dense_negative_budget:.3f} is {dense_to_objective_ratio:.2f}x the positive objective budget.")
        hard_constraints.append("Reduce absolute dense penalties or convert them into progress-delta feedback so longer useful attempts are not worse than early failure.")

    return {
        "objective_positive_budget": objective_budget,
        "terminal_failure_budget": terminal_failure_budget,
        "dense_negative_budget": dense_negative_budget,
        "dense_positive_budget": dense_positive_budget,
        "auxiliary_abs_budget": auxiliary_budget,
        "dominant_component": top,
        "dominance_ratio": dominance_ratio,
        "dense_to_objective_ratio": dense_to_objective_ratio,
        "risks": risks,
        "hints": hints,
        "hard_constraints": hard_constraints,
    }


def reward_payment_audit(evidence: Dict[str, Any], reward_schema: Dict[str, Any], reward_code: str) -> Dict[str, Any]:
    components = evidence.get("component_summary", {}).get("component_means", {}) or {}
    positive_components = [name for name, value in components.items() if _num(value) > 0 and name not in {"fitness_score", "total_reward"}]

    event_gated_terms = []
    repeatable_positive_terms = []
    raw_action_bonus_terms = []
    code_lower = reward_code.lower()

    for name in positive_components:
        lower = str(name).lower()
        if any(token in lower for token in SUCCESS_KEYS) or "terminated" in _code_window_for_key(code_lower, lower):
            event_gated_terms.append(name)
        else:
            repeatable_positive_terms.append(name)
        if any(token in lower for token in ACTION_HINT_KEYS):
            raw_action_bonus_terms.append(name)

    risks: List[str] = []
    hard_constraints: List[str] = []
    hints: List[str] = []

    if raw_action_bonus_terms:
        risks.append("raw_repeatable_action_bonus")
        hints.append("Reward payment audit: repeatable raw action/control bonuses detected: " + ", ".join(map(str, raw_action_bonus_terms)) + ".")
        hard_constraints.append("Do not simply increase raw per-step action bonuses; condition them on progress, phase, stability, or event milestones.")

    if repeatable_positive_terms and _num(evidence.get("primary_metrics", {}).get("success_like_terminal_rate")) <= 0.01:
        risks.append("repeatable_positive_without_success")
        hints.append("Reward payment audit: repeatable positive terms exist while success-like rate remains near zero.")
        hard_constraints.append("Check whether repeatable positive rewards can be farmed without completing the task.")

    schema_components = reward_schema.get("component_catalog") or reward_schema.get("components") or []
    declared_payment_modes = []
    for comp in schema_components:
        if isinstance(comp, dict) and comp.get("payment_mode"):
            declared_payment_modes.append({"id": comp.get("id"), "payment_mode": comp.get("payment_mode")})

    return {
        "repeatable_positive_terms": repeatable_positive_terms,
        "raw_action_bonus_terms": raw_action_bonus_terms,
        "event_gated_terms": event_gated_terms,
        "declared_payment_modes": declared_payment_modes,
        "risks": risks,
        "hints": hints,
        "hard_constraints": hard_constraints,
    }


def trajectory_phase_report(evidence: Dict[str, Any], task_model: Dict[str, Any] | None = None) -> Dict[str, Any]:
    behavior = evidence.get("behavior_summary", {}) or {}
    action_distribution = behavior.get("action_distribution", {}) or {}
    action_space_report = behavior.get("action_space_report", {}) or {}
    episodes = evidence.get("trajectory_examples", []) or []
    metrics = evidence.get("primary_metrics", {}) or {}

    near_pad = 0
    any_leg_contact = 0
    both_leg_contact = 0
    stable_touchdown = 0
    finite_final_state = 0
    final_states = []

    for ep in episodes:
        state = ep.get("final_state") or []
        if len(state) < 8:
            continue
        finite_final_state += 1
        x, y, vx, vy, angle = [_num(v) for v in state[:5]]
        leg1, leg2 = _num(state[6]), _num(state[7])
        is_near = abs(x) < 0.2 and abs(y) < 0.2
        has_leg = leg1 > 0.5 or leg2 > 0.5
        has_both = leg1 > 0.5 and leg2 > 0.5
        is_stable = is_near and has_both and abs(vx) < 0.2 and abs(vy) < 0.2 and abs(angle) < 0.2
        near_pad += int(is_near)
        any_leg_contact += int(has_leg)
        both_leg_contact += int(has_both)
        stable_touchdown += int(is_stable)
        final_states.append({
            "episode_id": ep.get("episode_id"),
            "near_pad": is_near,
            "any_leg_contact": has_leg,
            "both_leg_contact": has_both,
            "stable_touchdown": is_stable,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "angle": angle,
        })

    denom = max(1, finite_final_state)
    coverage_ratio = _num(action_space_report.get("coverage_ratio"))
    success_rate = _num(metrics.get("success_like_terminal_rate"))
    main_engine_usage = _num(action_distribution.get("2", action_distribution.get(2, 0.0)))

    phase_label = "unknown"
    if success_rate > 0.0:
        phase_label = "success_discovered"
    elif coverage_ratio >= 0.99 and main_engine_usage > 0.05:
        if any_leg_contact > 0 or near_pad > 0:
            phase_label = "near_ground_or_contact_unstable"
        else:
            phase_label = "controlled_descent_not_reaching_pad"
    elif action_space_report.get("unused_action_keys"):
        phase_label = "action_coverage_incomplete"
    elif _num(metrics.get("episode_length")) < 150:
        phase_label = "early_failure_before_phase_completion"

    hints: List[str] = []
    hard_constraints: List[str] = []
    if phase_label in {"near_ground_or_contact_unstable", "controlled_descent_not_reaching_pad"}:
        hints.append(f"Trajectory phase audit: action coverage is mostly solved but success is absent; failure phase appears to be {phase_label}.")
        hard_constraints.append("Do not keep optimizing only action coverage; add phase-specific guidance for controlled descent, near-ground stability, or touchdown.")

    return {
        "num_episode_examples": finite_final_state,
        "main_engine_usage": main_engine_usage,
        "action_coverage_ratio": coverage_ratio,
        "near_pad_rate": near_pad / denom,
        "any_leg_contact_rate": any_leg_contact / denom,
        "both_leg_contact_rate": both_leg_contact / denom,
        "stable_touchdown_rate": stable_touchdown / denom,
        "success_like_terminal_rate": success_rate,
        "failure_phase": phase_label,
        "final_state_digest": final_states[:5],
        "hints": hints,
        "hard_constraints": hard_constraints,
    }


def objective_signal_alignment_audit(evidence: Dict[str, Any]) -> Dict[str, Any]:
    components = evidence.get("component_summary", {}).get("component_means", {}) or {}
    success_rate = _num(evidence.get("primary_metrics", {}).get("success_like_terminal_rate"))
    objective_terms = {name: _num(value) for name, value in components.items() if any(token in str(name).lower() for token in SUCCESS_KEYS)}
    positive_objective_terms = {name: value for name, value in objective_terms.items() if value > 0}

    risks: List[str] = []
    hints: List[str] = []
    hard_constraints: List[str] = []
    if success_rate <= 0.01 and positive_objective_terms:
        risks.append("reward_success_signal_misaligned_with_evaluator")
        hints.append("Objective alignment audit: reward success/objective terms are positive while evaluator success_like_terminal_rate is near zero.")
        hard_constraints.append("Tighten success-like bonuses so they are event-gated and aligned with evaluator success, not merely proxy contact states.")

    return {
        "success_like_terminal_rate": success_rate,
        "positive_objective_terms": positive_objective_terms,
        "risks": risks,
        "hints": hints,
        "hard_constraints": hard_constraints,
    }


def recommend_search_mode(
    evidence: Dict[str, Any],
    scale_audit: Dict[str, Any],
    payment_audit: Dict[str, Any],
    phase_report: Dict[str, Any],
    objective_audit: Dict[str, Any],
) -> Dict[str, Any]:
    behavior = evidence.get("behavior_summary", {}) or {}
    action_space_report = behavior.get("action_space_report", {}) or {}
    unused = action_space_report.get("unused_action_keys") or []
    success_rate = _num(evidence.get("primary_metrics", {}).get("success_like_terminal_rate"))

    mode = "balanced_reward_revision"
    reason = "No single deterministic audit dominates; continue balanced expert revision."
    forbidden_edits: List[str] = []
    required_focus: List[str] = []

    if unused:
        mode = "action_coverage_repair"
        reason = "Some discrete actions are unused, so the next revision must first decide whether they are necessary."
        required_focus.append("Create reachable incentives for unused necessary actions without causing no-action collapse.")
    elif phase_report.get("failure_phase") in {"near_ground_or_contact_unstable", "controlled_descent_not_reaching_pad"} and success_rate <= 0.01:
        mode = "phase_guidance_after_action_coverage"
        reason = "Action coverage is mostly solved but success is absent; the bottleneck moved to trajectory phase completion."
        forbidden_edits.extend([
            "do not simply increase raw main-engine/action bonuses",
            "do not only reduce distance shaping",
        ])
        required_focus.extend([
            "Condition engine/control rewards on controlled descent and near-pad progress.",
            "Add near-ground stability or touchdown-phase guidance.",
        ])
    elif "raw_repeatable_action_bonus" in payment_audit.get("risks", []):
        mode = "payment_mode_rewrite"
        reason = "Reward contains repeatable raw action/control bonuses that may be farmed without task completion."
        forbidden_edits.append("do not increase repeatable per-step bonuses without event/progress gating")
        required_focus.append("Convert raw per-step bonuses into progress-conditioned, event-gated, one-shot, or diminishing signals.")
    elif scale_audit.get("dominance_ratio", 0.0) >= 3.0:
        mode = "scale_rebalance"
        reason = "One component dominates the episode-scale reward budget."
        required_focus.append("Reduce auxiliary dominance while preserving reachable task progress signals.")

    if objective_audit.get("risks"):
        required_focus.append("Align reward-defined success/objective bonuses with evaluator success-like behavior.")

    hints = [f"Search mode recommendation: {mode}. {reason}"]
    hard_constraints = [f"Follow next_search_mode={mode}: {reason}"]
    hard_constraints.extend(required_focus)
    hard_constraints.extend(["Forbidden next edit: " + item for item in forbidden_edits])

    return {
        "next_search_mode": mode,
        "reason": reason,
        "forbidden_edits": forbidden_edits,
        "required_focus": required_focus,
        "hints": hints,
        "hard_constraints": hard_constraints,
    }


def _code_window_for_key(code_lower: str, key_lower: str) -> str:
    idx = code_lower.find(key_lower)
    if idx < 0:
        return ""
    return code_lower[max(0, idx - 300): idx + 300]


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
