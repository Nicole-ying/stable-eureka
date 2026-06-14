"""
Serial single-candidate trainer using RL Zoo PPO config.
n_envs=16, 2M timesteps. Returns detailed diagnostics.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Reuse the masked env wrapper
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "reward_search"))
from env_wrapper import MaskedLunarLander, compile_reward_function

warnings.filterwarnings("ignore", category=UserWarning)

# RL Zoo PPO config for LunarLander-v2
PPO_CONFIG = dict(
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    learning_rate=3e-4,
    clip_range=0.2,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


def make_vec_env(reward_fn, n_envs: int = 16):
    """Create n_envs parallel masked LunarLander environments."""
    def _make():
        env = gym.make("LunarLander-v3")
        return MaskedLunarLander(env, reward_fn)
    return DummyVecEnv([_make for _ in range(n_envs)])


def train_and_eval(
    reward_code: str,
    run_name: str,
    work_dir: Path,
    total_timesteps: int = 2_000_000,
    n_envs: int = 16,
    eval_episodes: int = 20,
    device: str = "auto",
) -> dict:
    """
    Train PPO with RL Zoo config, then evaluate with detailed metrics.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / f"{run_name}.zip"

    reward_fn = compile_reward_function(reward_code)

    # --- Training with VecEnv ---
    train_env = make_vec_env(reward_fn, n_envs=n_envs)
    t0 = time.time()

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        **PPO_CONFIG,
        device=device,
    )
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0
    model.save(str(ckpt_path))
    train_env.close()

    # --- Evaluation (single env for detailed metrics) ---
    eval_env = gym.make("LunarLander-v3")
    eval_env = MaskedLunarLander(eval_env, reward_fn)

    generated_returns = []
    hidden_returns = []
    episode_lengths = []
    all_components: dict[str, list[float]] = {}
    all_actions = []
    landing_count = 0
    crash_count = 0
    timeout_count = 0
    final_states = []  # (x, y, vx, vy, angle, left_leg, right_leg) at episode end

    # Track per-step metrics for trajectory analysis
    all_step_distances = []  # distance from pad at each step
    all_step_velocities = []  # speed at each step
    all_step_angles = []

    for ep in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        gen_ret = 0.0
        hid_ret = 0.0
        ep_len = 0
        comp_sum: dict[str, float] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(int(action))

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1

            gen_ret += float(reward)
            hid_ret += float(info.get("_hidden_reward", 0.0))
            all_step_distances.append(float(np.sqrt(obs[0]**2 + obs[1]**2)))
            all_step_velocities.append(float(np.sqrt(obs[2]**2 + obs[3]**2)))
            all_step_angles.append(float(abs(obs[4])))

            for k, v in info.get("_components", {}).items():
                comp_sum[k] = comp_sum.get(k, 0.0) + float(v)

        generated_returns.append(gen_ret)
        hidden_returns.append(hid_ret)
        episode_lengths.append(ep_len)
        final_states.append({
            "x": float(obs[0]), "y": float(obs[1]),
            "vx": float(obs[2]), "vy": float(obs[3]),
            "angle": float(obs[4]),
            "left_leg": float(obs[6]), "right_leg": float(obs[7]),
        })

        for k, v in comp_sum.items():
            if k not in all_components:
                all_components[k] = []
            all_components[k].append(v)

        # Classify episode outcome
        final = final_states[-1]
        near_pad = abs(final["x"]) < 0.2 and final["y"] < 0.2
        both_legs = final["left_leg"] > 0.5 and final["right_leg"] > 0.5
        low_speed = abs(final["vx"]) < 0.5 and abs(final["vy"]) < 0.5
        upright = abs(final["angle"]) < 0.2

        if both_legs and near_pad and low_speed and upright:
            landing_count += 1
        elif terminated:
            crash_count += 1
        else:
            timeout_count += 1

    eval_env.close()

    # --- Aggregate ---
    component_means = {k: float(np.mean(v)) for k, v in all_components.items()}
    component_stds = {k: float(np.std(v)) for k, v in all_components.items()}
    action_arr = np.array(all_actions, dtype=float)

    result = {
        "run_name": run_name,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "train_time_seconds": round(train_time, 1),

        # Core scores
        "generated_return_mean": float(np.mean(generated_returns)),
        "generated_return_std": float(np.std(generated_returns)),
        "hidden_return_mean": float(np.mean(hidden_returns)),
        "hidden_return_std": float(np.std(hidden_returns)),
        "generated_hidden_gap": float(np.mean(generated_returns) - np.mean(hidden_returns)),

        # Episode outcomes
        "landing_rate": landing_count / eval_episodes,
        "crash_rate": crash_count / eval_episodes,
        "timeout_rate": timeout_count / eval_episodes,
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),

        # Behavior metrics
        "action_mean": float(np.mean(action_arr)),
        "action_std": float(np.std(action_arr)),
        "action_distribution": {
            str(a): float(np.mean(action_arr == a)) for a in range(4)
        },
        "step_distance_mean": float(np.mean(all_step_distances)),
        "step_velocity_mean": float(np.mean(all_step_velocities)),
        "step_angle_mean": float(np.mean(all_step_angles)),

        # Component returns
        "component_returns_mean": component_means,
        "component_returns_std": component_stds,

        # Final states for detailed analysis
        "final_states": final_states,
        "final_distance_mean": float(np.mean([
            np.sqrt(s["x"]**2 + s["y"]**2) for s in final_states
        ])),
    }

    # Save
    with open(work_dir / f"{run_name}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
