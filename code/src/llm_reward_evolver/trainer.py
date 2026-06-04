from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from .feedback import EpisodeSummary, TrainingStats, summarize_episodes
from .reward import RewardProgram
from .wrappers import CustomRewardWrapper


@dataclass
class TrainResult:
    model: Any
    stats: TrainingStats
    interrupted: bool = False
    error_message: str = ""


class ProgressTracker:
    def __init__(self, total_timesteps: int) -> None:
        self.total_timesteps = max(1, total_timesteps)
        self.current_timesteps = 0

    def update(self, timesteps: int) -> None:
        self.current_timesteps = max(0, timesteps)

    def progress(self) -> float:
        return min(1.0, self.current_timesteps / self.total_timesteps)


def describe_space(space: Any) -> str:
    return repr(space)


def make_env(env_name: str, seed: int) -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError("Training requires gymnasium. Install requirements.txt first.") from exc

    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def train_agent(
    env_name: str,
    reward_program: Optional[RewardProgram],
    total_timesteps: int,
    eval_episodes: int,
    target_score: float,
    seed: int,
    training_algorithm: str = "ppo",
    progress_provider: Optional[Callable[[], float]] = None,
) -> TrainResult:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError("Training requires stable-baselines3. Install requirements.txt first.") from exc

    tracker = ProgressTracker(total_timesteps)

    class _ProgressCallback(BaseCallback):
        def _on_step(self) -> bool:
            tracker.update(self.num_timesteps)
            return True

    env = make_env(env_name, seed)
    train_env = env
    model = None
    try:
        if reward_program is not None:
            provider = progress_provider or tracker.progress
            train_env = CustomRewardWrapper(env, reward_program, provider).unwrap()

        model = _build_model(training_algorithm, train_env, seed, PPO, DQN)
        model.learn(total_timesteps=total_timesteps, callback=_ProgressCallback())
        episodes = evaluate_model(model, env_name, eval_episodes, target_score, seed + 10_000)
        stats = summarize_episodes(episodes, target_score)
        if reward_program is not None:
            stats.reward_error_count = reward_program.error_count
            stats.reward_last_error = reward_program.last_error or ""
        return TrainResult(model=model, stats=stats)
    except Exception as exc:
        reward_error_count = reward_program.error_count if reward_program is not None else 0
        reward_last_error = reward_program.last_error if reward_program is not None else ""
        stats = TrainingStats(
            mean_eval_score=0.0,
            success_rate=0.0,
            mean_episode_length=0.0,
            trend="interrupted",
            converged=False,
            failure_mode="training interrupted before evaluation",
            interrupted=True,
            error_message=f"{type(exc).__name__}: {exc}",
            reward_error_count=reward_error_count,
            reward_last_error=reward_last_error or "",
        )
        return TrainResult(model=model, stats=stats, interrupted=True, error_message=stats.error_message)
    finally:
        train_env.close()


def _build_model(training_algorithm: str, train_env: Any, seed: int, ppo_cls: Any, dqn_cls: Any) -> Any:
    algorithm = training_algorithm.lower()
    if algorithm == "dqn":
        return dqn_cls(
            "MlpPolicy",
            train_env,
            learning_rate=4e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=128,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.25,
            exploration_final_eps=0.05,
            verbose=0,
            seed=seed,
            device="cpu",
        )
    if algorithm == "ppo":
        return ppo_cls(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
            device="cpu",
        )
    raise ValueError(f"Unsupported training_algorithm: {training_algorithm}")


def evaluate_model(
    model: Any,
    env_name: str,
    episodes: int,
    target_score: float,
    seed: int,
) -> List[EpisodeSummary]:
    env = make_env(env_name, seed)
    summaries: List[EpisodeSummary] = []
    for episode_id in range(episodes):
        obs, _info = env.reset(seed=seed + episode_id)
        done = False
        total_reward = 0.0
        length = 0
        start_x = _extract_x(obs)
        final_x = start_x
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            length += 1
            done = bool(terminated or truncated)
            final_x = _extract_x(obs)
        summaries.append(
            EpisodeSummary(
                reward=total_reward,
                length=length,
                success=total_reward >= target_score,
                start_x=start_x,
                final_x=final_x,
            )
        )
    env.close()
    return summaries


def inspect_env(env_name: str, seed: int) -> Tuple[str, str]:
    env = make_env(env_name, seed)
    observation = describe_space(env.observation_space)
    action = describe_space(env.action_space)
    known_description = known_env_description(env_name)
    if known_description:
        observation = f"{observation}\n{known_description}"
    env.close()
    return observation, action


def known_env_description(env_name: str) -> str:
    name = env_name.lower()
    if name.startswith("acrobot"):
        return (
            "Observation semantics for Acrobot-v1: obs = [cos(theta1), sin(theta1), "
            "cos(theta2), sin(theta2), thetaDot1, thetaDot2]. The end-effector height can be "
            "computed as height = -cos(theta1) - cos(theta1 + theta2), where "
            "cos(theta1 + theta2) = cos(theta1)*cos(theta2) - sin(theta1)*sin(theta2). "
            "Higher height is better; the task succeeds when the tip swings high enough. "
            "Useful reward shaping should encourage increasing tip height and useful angular "
            "velocity for swing-up, then stabilize near high tip position. Important: use "
            "absolute angular velocity magnitude abs(thetaDot1)+abs(thetaDot2), not signed "
            "thetaDot1+thetaDot2, because signed velocities can cancel. Reward the current "
            "high tip position as well as height progress; using only height difference can be too sparse. "
            "A robust dense reward usually includes: current high-tip score, height progress, "
            "angular-velocity magnitude for swing-up, a goal-height bonus, and a mild action cost. "
            "Keep original_reward as an always-on anchor. Do not manually clamp final reward to "
            "[-1, 1] inside compute_reward; framework-level clipping handles numerical safety. "
            "Action is a discrete integer in {0, 1, 2}; do not treat it as a vector."
        )
    if name.startswith("mountaincar"):
        return (
            "Observation semantics for MountainCar-v0: obs = [position, velocity]. "
            "The car must sometimes move left before moving right to build momentum. "
            "Useful reward shaping should encourage velocity magnitude, distance-to-goal reduction, "
            "and reaching the right goal position, not only immediate rightward movement."
        )
    if name.startswith("lunarlander"):
        return (
            "Observation semantics for LunarLander-v3: obs = [x_position, y_position, "
            "x_velocity, y_velocity, angle, angular_velocity, left_leg_contact, right_leg_contact]. "
            "Action is a discrete integer: 0=noop, 1=left orientation engine, 2=main engine, "
            "3=right orientation engine. The lander should approach x=0, y=0, reduce horizontal "
            "and vertical speed, keep angle near 0, and make both legs contact the ground. "
            "Useful dense reward components include distance-to-pad penalty, velocity penalty, "
            "angle/upright penalty, contact bonus for each leg, soft landing bonus, crash/tilt "
            "penalty, and fuel/action cost. Main engine action 2 should be mildly penalized for fuel "
            "but rewarded when it reduces dangerous downward velocity. Important sign constraints: "
            "distance, velocity, tilt, angular velocity, and fuel use are costs, so they should reduce "
            "reward. Contact and soft landing should increase reward. Keep original_reward as a bounded "
            "anchor so the true task objective is not overwritten. Do not multiply the whole reward by "
            "(1 - training_progress), because that removes the landing signal late in training. "
            "Do not use info keys such as fuel_used because the environment does not provide them."
        )
    if name.startswith("cartpole"):
        return (
            "Observation semantics for CartPole-v1: obs = [cart_position, cart_velocity, "
            "pole_angle, pole_angular_velocity]. Smaller absolute pole_angle and cart_position "
            "are better; longer survival is better."
        )
    return ""


def _extract_x(obs: Any) -> float:
    try:
        return float(obs[0])
    except Exception:
        return 0.0
