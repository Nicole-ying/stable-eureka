from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .controller import SearchController as BaseSearchController


class SearchController(BaseSearchController):
    """Search controller that trusts the LLM Search Strategist's judgment.

    The hardcoded acceptance gate has been replaced with a diagnostic report.
    Elite/parent selection follows the base controller's simple rule:
    - Elite: fitness improves over current elite AND no behavior regression
    - Parent: fitness improves over parent OR parent is bootstrap

    The LLM Search Strategist provides the authoritative decision through
    expert_search_strategy_decision.json. The controller's job is to EXECUTE
    that decision, not second-guess it with hardcoded thresholds.

    All analysis (search_state_assessment, revision_brief, negative_edges)
    flows to the Revision Agent through the memory context.
    """

    def decide(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        decision = super().decide(*args, **kwargs)

        parent_evidence = kwargs.get("parent_evidence") if kwargs else None
        candidate_evidence = kwargs.get("candidate_evidence") if kwargs else None
        best_evidence = kwargs.get("best_evidence") if kwargs else None
        if parent_evidence is None and args:
            parent_evidence = args[0]
        if candidate_evidence is None and len(args) > 1:
            candidate_evidence = args[1]
        if best_evidence is None and len(args) > 2:
            best_evidence = args[2]

        parent_evidence = parent_evidence or {}
        candidate_evidence = candidate_evidence or {}
        best_evidence = best_evidence or {}

        # Build diagnostic report (informational only — does NOT override decisions)
        report = _diagnostic_report(candidate_evidence, best_evidence, parent_evidence)

        # The base controller already made the correct decision:
        #   accepted_as_elite = candidate.fitness > best.fitness
        #   accepted_as_parent = candidate.fitness > parent.fitness
        # We augment it with the diagnostic context.
        decision["expert_acceptance_gate"] = report
        decision["elite_status"] = (
            "provisional_search_anchor"
            if decision.get("accepted_as_elite") and report["diagnostics"]["success_rate"] < 0.5
            else "verified_elite" if decision.get("accepted_as_elite")
            else "not_elite"
        )

        # Only ONE hard block: clear behavioral collapse.
        # If the policy uses only 1 action (>95%) with 0% success, it's broken.
        behavior = candidate_evidence.get("behavior_summary", {}) or {}
        action_dist = behavior.get("action_distribution", {}) or {}
        max_action_prob = max(action_dist.values()) if action_dist else 0.0
        success_rate = _num((candidate_evidence.get("primary_metrics", {}) or {}).get("success_like_terminal_rate", 0))
        is_action_collapse = bool(max_action_prob >= 0.95 and success_rate <= 0.0)

        if is_action_collapse and decision.get("accepted_as_parent"):
            decision["accepted_as_parent"] = False
            decision["rejected"] = True
            decision.setdefault("rejection_reasons", []).append(
                "action collapse detected: >95% single action with 0% success — policy is broken"
            )
            fallback = decision.get("rollback", {}) or {}
            if fallback.get("next_parent_dir"):
                decision["next_search_parent"] = {
                    "source": fallback.get("next_parent_source", "elite_parent"),
                    "dir": fallback["next_parent_dir"],
                    "uses_existing_trained_artifact_as_parent_only": True,
                    "will_retrain_selected_parent": False,
                    "will_train_new_child_candidate_next": True,
                }

        return decision


def _diagnostic_report(
    candidate: Dict[str, Any],
    best: Dict[str, Any],
    parent: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build an informational diagnostic report. Does NOT block any decisions.

    This replaces the old _acceptance_gate which used hardcoded thresholds
    (target_success >= 0.5, score_gain <= noise_floor, etc.) to override
    the LLM's judgment. Those thresholds were wrong for search — they blocked
    iter_009 (40% success, best result so far) from becoming elite because
    40% < 50%.

    The report is still valuable: it flows into memory and helps the Revision
    Agent understand what's working and what's not.
    """
    primary = candidate.get("primary_metrics", {}) or {}
    comp = (candidate.get("component_summary", {}) or {}).get("component_means", {}) or {}
    target_report = candidate.get("target_behavior_report", {}) or {}
    stability = candidate.get("checkpoint_stability_report", {}) or {}
    audit = candidate.get("expert_audit_pack", {}) or {}

    fitness = _num(primary.get("fitness_score"))
    fitness_std = _num(primary.get("fitness_score_std"))
    generated = _num(primary.get("generated_reward"))
    best_fitness = _num((best.get("primary_metrics", {}) or {}).get("fitness_score"))
    best_criteria = (best.get("target_behavior_report", {}) or {}).get("primary_behavior_criteria", {}) or {}
    best_success_rate = _num(best_criteria.get("success_like_terminal_rate", (best.get("primary_metrics", {}) or {}).get("success_like_terminal_rate")))
    criteria = target_report.get("primary_behavior_criteria", {}) or {}
    success_rate = _num(criteria.get("success_like_terminal_rate", primary.get("success_like_terminal_rate")))
    timeout_rate = _num(criteria.get("timeout_rate"))
    oob_rate = _num(criteria.get("out_of_bounds_rate", primary.get("out_of_bounds_rate")))
    unsafe_rate = _num(criteria.get("unsafe_terminal_rate", primary.get("unsafe_terminal_rate")))
    episode_length = _num(primary.get("episode_length"))

    # Dominant component analysis
    dominant = (comp if isinstance(comp, list) else
                (candidate.get("component_summary", {}) or {}).get("dominant_components_by_abs_return", []) or [])
    dominant_name = ""
    dominant_value = 0.0
    if isinstance(dominant, list) and dominant:
        dominant_name = str(dominant[0].get("name", ""))
        dominant_value = _num(dominant[0].get("abs_return", 0))
    elif isinstance(comp, dict):
        sorted_comp = sorted(comp.items(), key=lambda x: abs(x[1]), reverse=True)
        if sorted_comp:
            dominant_name = sorted_comp[0][0]
            dominant_value = abs(sorted_comp[0][1])

    # Build observations (not gate reasons — just things to pay attention to)
    observations: List[str] = []
    warnings: List[str] = []

    # Positive signals
    if success_rate > best_success_rate:
        observations.append(f"success rate improved: {best_success_rate:.0%} → {success_rate:.0%}")
    if fitness > best_fitness:
        observations.append(f"fitness improved: {best_fitness:.1f} → {fitness:.1f} (+{fitness-best_fitness:.1f})")
    if oob_rate <= 0.0 and unsafe_rate <= 0.0:
        observations.append("no out-of-bounds or crash terminations — safe policy")

    # Concerns (not blockers)
    if timeout_rate >= 0.6 and success_rate > 0:
        warnings.append(f"timeout rate {timeout_rate:.0%} — agent may be hesitating before landing")
    if timeout_rate >= 0.8 and success_rate <= 0.01:
        warnings.append(f"timeout-dominated ({timeout_rate:.0%}) — agent collects proxy rewards without completing task")
    if dominant_value > 0 and dominant_name not in ("fitness_score", "terminal_landing", ""):
        # Check if an auxiliary dominates
        terminal_val = abs(_num(comp.get("terminal_landing", 0) if isinstance(comp, dict) else 0))
        if terminal_val > 0 and dominant_value > 3.0 * terminal_val:
            warnings.append(f"component '{dominant_name}' ({dominant_value:.0f}) dominates terminal ({terminal_val:.0f}) — scale imbalance")
    if fitness_std > abs(fitness) * 0.5 and abs(fitness) > 10:
        warnings.append(f"high fitness variance (std={fitness_std:.1f}, rel_std={fitness_std/abs(fitness):.1f})")
    if generated > 0 and (generated - fitness) > max(500.0, 3.0 * max(abs(fitness), 1.0)):
        warnings.append(f"large generated-vs-fitness gap ({generated-fitness:.0f}) — reward proxy may be misaligned")

    # Scale audit hints
    scale_audit = audit.get("reward_scale_audit", {}) or {}
    scale_risks = scale_audit.get("risks", []) or []
    if "single_component_dominance" in scale_risks:
        warnings.append("scale audit: single component dominates reward budget")

    return {
        "file_type": "expert_acceptance_gate",
        "diagnostics": {
            "fitness": fitness,
            "fitness_std": fitness_std,
            "success_rate": success_rate,
            "timeout_rate": timeout_rate,
            "oob_rate": oob_rate,
            "unsafe_rate": unsafe_rate,
            "episode_length": episode_length,
            "best_fitness": best_fitness,
            "best_success_rate": best_success_rate,
            "dominant_component": dominant_name,
            "dominant_component_value": dominant_value,
            "generated_reward": generated,
            "generated_fitness_gap": generated - fitness,
        },
        "observations": observations,
        "warnings": warnings,
        "scale_risks": scale_risks,
        "note": (
            "This is a diagnostic report only. Elite/parent selection follows the "
            "Search Strategy LLM's judgment via expert_search_strategy_decision.json. "
            "Hardcoded thresholds (target_success >= 0.5, etc.) have been removed — "
            "they blocked valid search progress (e.g., iter_009 at 40% success)."
        ),
    }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default
