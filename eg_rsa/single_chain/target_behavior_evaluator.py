from __future__ import annotations

import math
from typing import Any, Dict, List


def build_target_behavior_report(
    metrics: Dict[str, Any],
    trajectories: List[Dict[str, Any]],
    max_episode_steps: int | None = None,
    fitness_score_auxiliary: float | None = None,
) -> Dict[str, Any]:
    """Evaluate whether a reward function induces the intended task behavior.

    This report is not another reward score. It is a behavior contract diagnostic:
    fitness_score can help rank candidates, but the primary question is whether
    the trained policy actually terminates in a task-successful way rather than
    exploiting dense proxy payments or timeout behavior. Fitness uncertainty is
    recorded so the expert controller can avoid over-trusting noisy candidates.
    """
    success_rate = _num(metrics.get("success_like_terminal_rate"))
    unsafe_rate = _num(metrics.get("unsafe_terminal_rate"))
    out_of_bounds_rate = _num(metrics.get("out_of_bounds_rate"))
    episode_length = _num(metrics.get("episode_length"))
    episode_length_std = _num(metrics.get("episode_length_std"))
    generated_reward = _num(metrics.get("reward"))
    generated_reward_std = _num(metrics.get("reward_std"))
    fitness_score = _num(metrics.get("fitness_score", fitness_score_auxiliary))
    fitness_score_std = _num(metrics.get("fitness_score_std"))
    num_eval_episodes = len(trajectories)
    timeout_rate = _terminal_rate(trajectories, "timeout_or_truncated")

    near_pad_timeout_rate = 0.0
    both_legs_timeout_rate = 0.0
    stable_contact_timeout_rate = 0.0
    if trajectories:
        near_pad_timeout_rate = sum(1.0 for t in trajectories if _is_timeout(t) and _near_pad(t.get("final_state"))) / len(trajectories)
        both_legs_timeout_rate = sum(1.0 for t in trajectories if _is_timeout(t) and _both_legs(t.get("final_state"))) / len(trajectories)
        stable_contact_timeout_rate = sum(1.0 for t in trajectories if _is_timeout(t) and _stable_contact(t.get("final_state"))) / len(trajectories)

    fitness_sem = fitness_score_std / math.sqrt(max(1, num_eval_episodes))
    fitness_relative_std = fitness_score_std / max(abs(fitness_score), 1.0)
    high_fitness_uncertainty = bool(fitness_score_std > 50.0 or fitness_relative_std > 0.35)
    low_sample_warning = bool(num_eval_episodes < 10)
    evidence_confidence = "medium"
    if high_fitness_uncertainty or low_sample_warning:
        evidence_confidence = "low"
    if num_eval_episodes >= 20 and not high_fitness_uncertainty:
        evidence_confidence = "high"

    max_steps = float(max_episode_steps or 0)
    final_reaches_time_limit = bool(max_steps and episode_length >= 0.9 * max_steps)
    dense_proxy_timeout_risk = bool(timeout_rate >= 0.5 and success_rate <= 0.2 and generated_reward > 0.0)
    near_pad_timeout_risk = bool(near_pad_timeout_rate >= 0.4 or stable_contact_timeout_rate >= 0.4)

    target_success = bool(success_rate >= 0.5 and timeout_rate <= 0.4 and unsafe_rate <= 0.2 and out_of_bounds_rate <= 0.2)
    quality_label = "target_success_stable_enough" if target_success else "target_behavior_gap"
    behavior_gaps: List[str] = []
    required_reward_fixes: List[str] = []

    if success_rate < 0.5:
        behavior_gaps.append("Final policy does not reliably terminate with task success.")
        required_reward_fixes.append("Increase the distinction between true terminal success and proxy progress states.")
    if timeout_rate > 0.4:
        behavior_gaps.append("Final policy often reaches the time limit instead of completing the task.")
        required_reward_fixes.append("Penalize or stop paying proxy rewards for timeout-like behavior, especially near the target state.")
    if near_pad_timeout_risk:
        behavior_gaps.append("Policy reaches a near-target or stable-contact state but does not complete the task terminally.")
        required_reward_fixes.append("Make near-target/contact rewards diminishing, capped, or conditional on terminal completion.")
    if dense_proxy_timeout_risk:
        behavior_gaps.append("Generated reward remains attractive during non-success timeout behavior.")
        required_reward_fixes.append("Reduce farmable dense rewards and align payments with task completion phases.")
    if unsafe_rate > 0.2 or out_of_bounds_rate > 0.2:
        behavior_gaps.append("Unsafe or out-of-bounds terminations remain too frequent.")
        required_reward_fixes.append("Add clearer negative feedback for unsafe terminal behavior without overwhelming exploration.")
    if high_fitness_uncertainty:
        behavior_gaps.append("Auxiliary fitness has high episode-level variance; do not over-trust small score differences.")
        required_reward_fixes.append("Prefer reward edits that improve success/timeout behavior rather than chasing small noisy fitness gains.")
    if not behavior_gaps:
        behavior_gaps.append("No obvious target behavior gap from final evaluation; use fitness and robustness as auxiliary evidence.")

    return {
        "file_type": "target_behavior_report",
        "target_success": target_success,
        "quality_label": quality_label,
        "evidence_confidence": evidence_confidence,
        "primary_behavior_criteria": {
            "success_like_terminal_rate": success_rate,
            "timeout_rate": timeout_rate,
            "unsafe_terminal_rate": unsafe_rate,
            "out_of_bounds_rate": out_of_bounds_rate,
        },
        "auxiliary_scores": {
            "fitness_score": fitness_score,
            "fitness_score_std": fitness_score_std,
            "fitness_score_sem": fitness_sem,
            "generated_reward": generated_reward,
            "generated_reward_std": generated_reward_std,
            "episode_length": episode_length,
            "episode_length_std": episode_length_std,
            "num_eval_episodes": num_eval_episodes,
        },
        "auxiliary_score_stability": {
            "fitness_score_std": fitness_score_std,
            "fitness_score_sem": fitness_sem,
            "fitness_relative_std": fitness_relative_std,
            "high_fitness_uncertainty": high_fitness_uncertainty,
            "low_sample_warning": low_sample_warning,
        },
        "behavior_diagnostics": {
            "final_reaches_time_limit": final_reaches_time_limit,
            "near_pad_timeout_rate": near_pad_timeout_rate,
            "both_legs_contact_timeout_rate": both_legs_timeout_rate,
            "stable_contact_timeout_rate": stable_contact_timeout_rate,
            "dense_proxy_timeout_risk": dense_proxy_timeout_risk,
            "near_pad_timeout_risk": near_pad_timeout_risk,
        },
        "target_behavior_gaps": _dedupe(behavior_gaps),
        "required_reward_fixes": _dedupe(required_reward_fixes),
        "interpretation": (
            "Use this report as the main reward-quality evidence. Fitness is an auxiliary evaluator with uncertainty; "
            "generated reward is only a diagnostic of what the policy learned to collect."
        ),
    }


def build_checkpoint_stability_report(evals: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize whether training behavior is stable without selecting a peak checkpoint."""
    fitness_values = _num_list(evals.get("fitness_score"))
    fitness_std_values = _num_list(evals.get("fitness_score_std"))
    success_values = _num_list(evals.get("success_like_terminal_rate"))
    reward_values = _num_list(evals.get("reward"))
    timesteps = _num_list(evals.get("timesteps"))

    if not fitness_values:
        return {"file_type": "checkpoint_stability_report", "available": False}

    best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
    last_idx = len(fitness_values) - 1
    best_fitness = fitness_values[best_idx]
    final_fitness = fitness_values[last_idx]
    best_success = success_values[best_idx] if best_idx < len(success_values) else 0.0
    final_success = success_values[last_idx] if last_idx < len(success_values) else 0.0
    best_reward = reward_values[best_idx] if best_idx < len(reward_values) else 0.0
    final_reward = reward_values[last_idx] if last_idx < len(reward_values) else 0.0
    best_timestep = timesteps[best_idx] if best_idx < len(timesteps) else None
    final_timestep = timesteps[last_idx] if last_idx < len(timesteps) else None
    gap = best_fitness - final_fitness
    success_drop = best_success - final_success
    last_k = min(3, len(fitness_values))
    last_k_fitness = fitness_values[-last_k:]
    last_k_success = success_values[-last_k:] if len(success_values) >= last_k else []
    last_k_mean = _mean(last_k_fitness)
    last_k_std = _std(last_k_fitness)
    last_k_success_mean = _mean(last_k_success)
    final_fitness_std = fitness_std_values[last_idx] if last_idx < len(fitness_std_values) else 0.0

    transient_peak_risk = bool(gap > 50.0 or success_drop >= 0.4)
    reward_behavior_divergence = bool(final_reward > best_reward and final_fitness < best_fitness - 30.0)
    final_uncertainty_risk = bool(final_fitness_std > 50.0 or final_fitness_std / max(abs(final_fitness), 1.0) > 0.35)
    unstable_last_k_risk = bool(last_k >= 3 and last_k_std > 50.0)

    hard_constraints = []
    if transient_peak_risk:
        hard_constraints.extend([
            "Do not accept a reward as elite based on a transient peak checkpoint.",
            "Reward quality should be judged by final behavior, last-eval stability, and target behavior success; fitness is auxiliary evidence.",
        ])
    if final_uncertainty_risk or unstable_last_k_risk:
        hard_constraints.append("Do not over-trust small fitness differences when fitness variance is high; require behavior improvement or more robust evaluation.")

    return {
        "file_type": "checkpoint_stability_report",
        "available": True,
        "best_checkpoint_is_diagnostic_only": True,
        "best_fitness": best_fitness,
        "final_fitness": final_fitness,
        "final_fitness_std": final_fitness_std,
        "best_final_fitness_gap": gap,
        "best_success_like_terminal_rate": best_success,
        "final_success_like_terminal_rate": final_success,
        "success_rate_drop": success_drop,
        "best_generated_reward": best_reward,
        "final_generated_reward": final_reward,
        "best_timestep": best_timestep,
        "final_timestep": final_timestep,
        "last_k_eval": {
            "k": last_k,
            "fitness_mean": last_k_mean,
            "fitness_std": last_k_std,
            "success_like_terminal_rate_mean": last_k_success_mean,
        },
        "transient_peak_risk": transient_peak_risk,
        "reward_behavior_divergence": reward_behavior_divergence,
        "final_uncertainty_risk": final_uncertainty_risk,
        "unstable_last_k_risk": unstable_last_k_risk,
        "hard_constraints": hard_constraints,
    }


def _terminal_rate(trajectories: List[Dict[str, Any]], label: str) -> float:
    if not trajectories:
        return 0.0
    return sum(1.0 for t in trajectories if t.get("terminal_classification") == label) / len(trajectories)


def _is_timeout(traj: Dict[str, Any]) -> bool:
    return traj.get("terminal_classification") == "timeout_or_truncated"


def _near_pad(final_state: Any) -> bool:
    try:
        state = list(final_state or [])
        return len(state) >= 2 and abs(float(state[0])) < 0.1 and abs(float(state[1])) < 0.05
    except Exception:
        return False


def _both_legs(final_state: Any) -> bool:
    try:
        state = list(final_state or [])
        return len(state) >= 8 and float(state[6]) > 0.5 and float(state[7]) > 0.5
    except Exception:
        return False


def _stable_contact(final_state: Any) -> bool:
    try:
        state = list(final_state or [])
        if len(state) < 8:
            return False
        return _near_pad(state) and _both_legs(state) and abs(float(state[2])) < 0.05 and abs(float(state[3])) < 0.05 and abs(float(state[4])) < 0.2
    except Exception:
        return False


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _num_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            pass
    return out


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
