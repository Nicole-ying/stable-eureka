from __future__ import annotations

from textwrap import dedent
from typing import Optional


def build_initial_prompt(
    env_name: str,
    task_description: str,
    observation_description: str,
    action_description: str,
    reward_structure: str = "hrdc",
) -> str:
    structure_rule = _structure_rule(reward_structure)
    env_rule = _environment_reward_rule(env_name)
    return dedent(
        f"""
        You are an expert reinforcement learning reward engineer.
        Generate one Python function named compute_reward with this exact signature:

        def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
            ...

        Requirements:
        - Return a single float reward.
        - Do not import modules.
        - Use only obs, action, next_obs, original_reward, info, and training_progress.
        - {structure_rule}
        - Preserve the original task objective: include original_reward as a bounded anchor term.
        - Keep the reward numerically stable.
        - Do not manually clamp the reward inside compute_reward; the training framework performs final clipping.
        - Do not multiply original_reward by (1 - training_progress); the task anchor must remain active through training.
        - Output only Python code.
        {env_rule}

        Environment: {env_name}
        Observation interface: {observation_description}
        Action interface: {action_description}
        Task: {task_description}
        """
    ).strip()


def build_refine_prompt(
    env_name: str,
    task_description: str,
    current_code: str,
    feedback: str,
    previous_best_code: Optional[str] = None,
    reward_structure: str = "hrdc",
) -> str:
    best_block = f"\nPrevious best reward code:\n{previous_best_code}\n" if previous_best_code else ""
    structure_rule = _structure_rule(reward_structure)
    env_rule = _environment_reward_rule(env_name)
    return dedent(
        f"""
        You are improving an RL reward function using FDRE diagnostic feedback.
        Modify the current function conservatively. Keep the same signature:

        def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
            ...

        Rules:
        - Output only Python code.
        - Do not import modules.
        - {structure_rule}
        - Fix the failure mode described by the feedback.
        - Preserve the original task objective: keep original_reward as a bounded anchor term.
        - Avoid large reward scale changes unless feedback clearly requires it.
        - Prefer conservative edits: keep useful components from previous_best_code and change only the weakest component.
        - If previous_best_code exists, do not rewrite the reward from scratch.
        - Do not manually clamp the reward inside compute_reward; the training framework performs final clipping.
        - Do not multiply original_reward by (1 - training_progress); the task anchor must remain active through training.
        - Avoid undefined variables. Before returning, check that every variable has been assigned.
        - For discrete actions, treat action as a scalar integer, not an array.
        {env_rule}

        Environment: {env_name}
        Task: {task_description}

        Current reward code:
        {current_code}
        {best_block}
        FDRE feedback / diagnostic report:
        {feedback}
        """
    ).strip()


def _structure_rule(reward_structure: str) -> str:
    if reward_structure == "static":
        return (
            "Use decomposed reward components with fixed weights only; do not change weights "
            "based on training_progress or state. This is an ablation without dynamic weighting."
        )
    if reward_structure == "flat":
        return "Use a single flat reward expression without HRDC decomposition. This is an ablation."
    return (
        "Use an HRDC-style structure: decompose the task into reward components, then combine "
        "them with stage weights based on training_progress."
    )


def _environment_reward_rule(env_name: str) -> str:
    name = env_name.lower()
    if name.startswith("lunarlander"):
        return (
            "\n        LunarLander-specific rules:\n"
            "        - Bad quantities must reduce reward: distance, speed, tilt, angular speed, crash risk, and fuel use are negative costs.\n"
            "        - Do not create positive variables named penalty; a penalty must be subtracted or negative.\n"
            "        - Do not multiply the whole reward by (1 - training_progress); late training still needs a strong landing signal.\n"
            "        - Do not use unavailable keys such as info['fuel_used']; infer fuel cost from the discrete action.\n"
            "        - A good state near x=0, y=0 with low velocity, upright angle, and both leg contacts must score higher than a far, fast, tilted state.\n"
        )
    if name.startswith("acrobot"):
        return (
            "\n        Acrobot-specific rules:\n"
            "        - Reward current high tip position and useful swing-up progress; one-step progress alone is too sparse.\n"
            "        - Use angular velocity magnitude abs(thetaDot1) + abs(thetaDot2), not signed velocity sums.\n"
            "        - A robust pattern is original_reward + high_tip_position + height_progress + small angular-velocity magnitude + goal-height bonus - mild action cost.\n"
            "        - Do not hard clamp the final reward to [-1, 1]; that erases useful height and goal-bonus differences.\n"
        )
    if name.startswith("mountaincar"):
        return (
            "\n        MountainCar-specific rules:\n"
            "        - Reward momentum as absolute velocity because the car must first move both left and right.\n"
            "        - Do not reward only immediate rightward movement.\n"
        )
    return ""
