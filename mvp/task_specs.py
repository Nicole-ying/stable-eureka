from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    objective: str
    obs_hint: str
    action_hint: str
    success_hint: str
    failure_hint: str
    judge_rubric: str


TASK_SPECS: dict[str, TaskSpec] = {
    "LunarLander-v3": TaskSpec(
        env_id="LunarLander-v3",
        objective="Land the craft softly at the center pad and remain stable.",
        obs_hint="obs includes position/velocity/angle/angular velocity/leg contacts.",
        action_hint="discrete engine controls; avoid wasteful thrust.",
        success_hint="upright low-speed touchdown near center with low oscillation.",
        failure_hint="hard crash, drifting far from pad, spinning, fuel-wasting jitter.",
        judge_rubric=(
            "Prioritize touchdown success, then smoothness, then energy efficiency. "
            "Penalize crash, unstable attitude, and persistent oscillation."
        ),
    ),
    "BipedalWalker-v3": TaskSpec(
        env_id="BipedalWalker-v3",
        objective="Walk forward robustly with stable gait and low energy waste.",
        obs_hint="obs includes hull angle/velocity, joint states, lidar terrain info.",
        action_hint="continuous torques for hip/knee joints.",
        success_hint="long forward progress, balanced torso, rhythmic gait.",
        failure_hint="falling, dragging, severe shaking, no forward progress.",
        judge_rubric=(
            "Prioritize forward progress and not falling, then smooth gait and low jitter."
        ),
    ),
}


def get_task_spec(env_id: str) -> TaskSpec:
    return TASK_SPECS.get(
        env_id,
        TaskSpec(
            env_id=env_id,
            objective="Solve the environment with stable and robust behavior.",
            obs_hint="Infer useful state factors from observation vectors.",
            action_hint="Use action penalties only when clearly beneficial.",
            success_hint="Consistent completion and stable control.",
            failure_hint="Instability, oscillation, and frequent failure endings.",
            judge_rubric="Score by task completion, smoothness, robustness, and safety.",
        ),
    )
