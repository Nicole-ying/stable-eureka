from __future__ import annotations

from typing import Any, Dict, List


def build_target_behavior_contract(context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a lightweight target behavior contract for behavior-quality audits.

    The contract is a structured expert prior about what success, failure, and
    dangerous local optima mean for the current task. It keeps environment-specific
    behavior checks outside the generic evaluator.
    """
    context = context or {}
    env_name = _environment_name(context).lower()
    if "lunar" in env_name and "lander" in env_name:
        return _lunar_lander_contract(context)
    if "cartpole" in env_name or "cart_pole" in env_name:
        return _cartpole_contract(context)
    if "bipedal" in env_name or "walker" in env_name:
        return _bipedal_walker_contract(context)
    return _generic_contract(context)


def assess_contract_specific_behavior(
    contract: Dict[str, Any],
    trajectories: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    max_episode_steps: int | None = None,
) -> Dict[str, Any]:
    name = str(contract.get("contract_name", "GenericTargetBehaviorContract"))
    if name == "LunarLanderTargetBehaviorContract":
        return _assess_lunar_lander(trajectories, metrics, max_episode_steps)
    if name == "CartPoleTargetBehaviorContract":
        return _assess_cartpole(trajectories, metrics, max_episode_steps)
    if name == "BipedalWalkerTargetBehaviorContract":
        return _assess_bipedal_walker(trajectories, metrics, max_episode_steps)
    return _assess_generic(trajectories, metrics, max_episode_steps)


def _lunar_lander_contract(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_behavior_contract",
        "contract_name": "LunarLanderTargetBehaviorContract",
        "environment_name": _environment_name(context),
        "success_terminal": [
            "safe_landing_terminal or success_like_terminal is observed",
            "episode terminates by task completion, not by time limit",
        ],
        "failure_terminal": ["crash_terminal", "unsafe_terminal", "out_of_bounds_terminal", "timeout_terminal"],
        "proxy_success_risks": [
            "near-pad state without terminal safe landing",
            "both-leg contact or stable contact that continues until timeout",
            "dense distance/velocity/contact reward farming without task completion",
        ],
        "forbidden_local_optima": [
            "near_pad_timeout",
            "stable_contact_timeout",
            "hovering_or_settling_until_time_limit",
        ],
        "minimum_behavior_evidence": [
            "success_like_terminal_rate",
            "timeout_rate",
            "unsafe_terminal_rate",
            "final_state samples",
            "trajectory terminal_classification",
        ],
        "state_assumptions": {
            "state_format": "[x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]",
            "near_pad_threshold": {"abs_x": 0.1, "abs_y": 0.05},
            "stable_contact_threshold": {"abs_vx": 0.05, "abs_vy": 0.05, "abs_angle": 0.2},
        },
    }


def _cartpole_contract(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_behavior_contract",
        "contract_name": "CartPoleTargetBehaviorContract",
        "environment_name": _environment_name(context),
        "success_terminal": ["long episode survival up to the task horizon"],
        "failure_terminal": ["pole angle or cart position violates environment limits"],
        "proxy_success_risks": ["standing still under a shaped proxy while failing survival", "over-regularized action policy"],
        "forbidden_local_optima": ["early_balanced_failure", "single_action_collapse"],
        "minimum_behavior_evidence": ["episode_length", "episode_length_std", "success_like_terminal_rate", "action_distribution"],
        "state_assumptions": {},
    }


def _bipedal_walker_contract(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_behavior_contract",
        "contract_name": "BipedalWalkerTargetBehaviorContract",
        "environment_name": _environment_name(context),
        "success_terminal": ["sustained forward locomotion without falling"],
        "failure_terminal": ["falling", "stalling", "early termination"],
        "proxy_success_risks": ["standing still", "energy-minimizing no-motion", "fall-forward reward farming"],
        "forbidden_local_optima": ["no_motion_energy_saving", "early_fall", "unstable_hopping_without_progress"],
        "minimum_behavior_evidence": ["fitness_score", "episode_length", "action_distribution", "trajectory phase evidence"],
        "state_assumptions": {},
    }


def _generic_contract(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_behavior_contract",
        "contract_name": "GenericTargetBehaviorContract",
        "environment_name": _environment_name(context),
        "success_terminal": ["environment-defined success-like terminal behavior when available"],
        "failure_terminal": ["unsafe terminal", "out of bounds", "timeout without task completion"],
        "proxy_success_risks": ["dense proxy reward collection without task completion", "single-action collapse", "timeout local optimum"],
        "forbidden_local_optima": ["timeout_without_success", "proxy_reward_farming", "single_action_collapse"],
        "minimum_behavior_evidence": ["success_like_terminal_rate", "unsafe_terminal_rate", "episode_length", "action_distribution"],
        "state_assumptions": {},
    }


def _assess_lunar_lander(trajectories: List[Dict[str, Any]], metrics: Dict[str, Any], max_episode_steps: int | None) -> Dict[str, Any]:
    near_pad_timeout_rate = 0.0
    both_legs_timeout_rate = 0.0
    stable_contact_timeout_rate = 0.0
    if trajectories:
        near_pad_timeout_rate = sum(1.0 for t in trajectories if _is_timeout(t) and _near_pad(t.get("final_state"))) / len(trajectories)
        both_legs_timeout_rate = sum(1.0 for t in trajectories if _is_timeout(t) and _both_legs(t.get("final_state"))) / len(trajectories)
        stable_contact_timeout_rate = sum(1.0 for t in trajectories if _is_timeout(t) and _stable_contact(t.get("final_state"))) / len(trajectories)
    return {
        "contract_specific_metrics": {
            "near_pad_timeout_rate": near_pad_timeout_rate,
            "both_legs_contact_timeout_rate": both_legs_timeout_rate,
            "stable_contact_timeout_rate": stable_contact_timeout_rate,
        },
        "contract_risks": {
            "near_pad_timeout_risk": bool(near_pad_timeout_rate >= 0.4 or stable_contact_timeout_rate >= 0.4),
        },
        "contract_behavior_gaps": [
            "Policy reaches a near-target or stable-contact state but does not complete the task terminally."
        ] if near_pad_timeout_rate >= 0.4 or stable_contact_timeout_rate >= 0.4 else [],
        "contract_required_fixes": [
            "Make near-target/contact rewards diminishing, capped, or conditional on terminal completion."
        ] if near_pad_timeout_rate >= 0.4 or stable_contact_timeout_rate >= 0.4 else [],
    }


def _assess_cartpole(trajectories: List[Dict[str, Any]], metrics: Dict[str, Any], max_episode_steps: int | None) -> Dict[str, Any]:
    max_steps = float(max_episode_steps or 0)
    episode_length = _num(metrics.get("episode_length"))
    short_survival_risk = bool(max_steps and episode_length < 0.6 * max_steps)
    return {
        "contract_specific_metrics": {"survival_ratio": episode_length / max_steps if max_steps else 0.0},
        "contract_risks": {"short_survival_risk": short_survival_risk},
        "contract_behavior_gaps": ["Policy does not survive long enough to satisfy the task horizon."] if short_survival_risk else [],
        "contract_required_fixes": ["Strengthen early recoverability and stable balancing signals without collapsing action diversity."] if short_survival_risk else [],
    }


def _assess_bipedal_walker(trajectories: List[Dict[str, Any]], metrics: Dict[str, Any], max_episode_steps: int | None) -> Dict[str, Any]:
    return _assess_generic(trajectories, metrics, max_episode_steps)


def _assess_generic(trajectories: List[Dict[str, Any]], metrics: Dict[str, Any], max_episode_steps: int | None) -> Dict[str, Any]:
    max_steps = float(max_episode_steps or 0)
    episode_length = _num(metrics.get("episode_length"))
    success_rate = _num(metrics.get("success_like_terminal_rate"))
    timeout_like = bool(max_steps and episode_length >= 0.9 * max_steps and success_rate < 0.2)
    return {
        "contract_specific_metrics": {"timeout_like_episode_length": timeout_like},
        "contract_risks": {"timeout_without_success_risk": timeout_like},
        "contract_behavior_gaps": ["Policy tends toward timeout without success-like completion."] if timeout_like else [],
        "contract_required_fixes": ["Reduce dense proxy payments that remain collectible during timeout behavior."] if timeout_like else [],
    }


def _environment_name(context: Dict[str, Any]) -> str:
    for key in ["environment_name", "env_name", "name"]:
        value = context.get(key)
        if isinstance(value, str) and value:
            return value
    task_goal = context.get("task_goal") or context.get("primary_objective") or ""
    if isinstance(task_goal, dict):
        for value in task_goal.values():
            if isinstance(value, str) and "lunar" in value.lower():
                return value
    if isinstance(task_goal, str):
        return task_goal
    return "unknown_environment"


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
