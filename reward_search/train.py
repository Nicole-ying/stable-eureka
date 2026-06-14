"""
Train a PPO agent with a custom reward function on LunarLander-v3.
Tracks detailed metrics for reward function evaluation.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from env_wrapper import MaskedLunarLander, compile_reward_function, make_env

# Suppress SB3 GPU warning
warnings.filterwarnings("ignore", category=UserWarning)


def train_and_eval(
    reward_code: str,
    run_name: str,
    work_dir: Path,
    total_timesteps: int = 30_000,
    eval_episodes: int = 10,
    device: str = "auto",
    # PPO hyperparams (RL Zoo LunarLander style)
    n_steps: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 4,
    gae_lambda: float = 0.98,
    gamma: float = 0.999,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> dict:
    """
    Train PPO with the given reward function, then evaluate.
    Returns a dict with all metrics.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / f"{run_name}.zip"

    # Compile reward function
    reward_fn = compile_reward_function(reward_code)

    # --- Training ---
    train_env = make_env(reward_fn)
    t0 = time.time()

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
    )
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0
    model.save(str(ckpt_path))
    train_env.close()

    # --- Evaluation ---
    eval_env = make_env(reward_fn)
    generated_returns = []
    hidden_returns = []
    episode_lengths = []
    all_components: dict[str, list[float]] = {}
    all_actions = []
    all_positions = []      # (x, y) trajectory
    landing_count = 0
    crash_count = 0

    for ep in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        gen_ret = 0.0
        hid_ret = 0.0
        ep_len = 0
        comp_sum: dict[str, float] = {}
        ep_positions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(int(action))

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)

            gen_ret += float(reward)
            hid_ret += float(info.get("_hidden_reward", 0.0))
            ep_len += 1
            ep_positions.append((float(obs[0]), float(obs[1])))

            for k, v in info.get("_components", {}).items():
                comp_sum[k] = comp_sum.get(k, 0.0) + float(v)

        generated_returns.append(gen_ret)
        hidden_returns.append(hid_ret)
        episode_lengths.append(ep_len)
        all_positions.append(ep_positions)

        for k, v in comp_sum.items():
            all_components.setdefault(k, []).append(v)

        # Landing detection: both legs on ground near pad
        final_obs = obs  # last observation
        near_pad = abs(final_obs[0]) < 0.2 and abs(final_obs[1]) < 0.2
        both_legs = final_obs[6] > 0.5 and final_obs[7] > 0.5
        low_speed = abs(final_obs[2]) < 0.5 and abs(final_obs[3]) < 0.5
        upright = abs(final_obs[4]) < 0.2

        if both_legs and near_pad and low_speed and upright:
            landing_count += 1
        elif terminated and not (both_legs and near_pad):
            crash_count += 1

    eval_env.close()

    # --- Aggregate metrics ---
    component_means = {
        k: float(np.mean(v)) for k, v in all_components.items()
    }

    action_arr = np.array(all_actions, dtype=float)

    # Distance from pad over time (average final distance)
    final_distances = [
        np.sqrt(p[-1][0]**2 + p[-1][1]**2) if p else 9.0
        for p in all_positions
    ]

    result = {
        "run_name": run_name,
        "total_timesteps": total_timesteps,
        "train_time_seconds": round(train_time, 1),
        "eval_episodes": eval_episodes,

        # Core metrics
        "generated_return_mean": float(np.mean(generated_returns)),
        "generated_return_std": float(np.std(generated_returns)),
        "hidden_return_mean": float(np.mean(hidden_returns)),
        "hidden_return_std": float(np.std(hidden_returns)),
        "generated_hidden_gap": float(np.mean(generated_returns) - np.mean(hidden_returns)),

        # Behavior metrics
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "landing_rate": landing_count / eval_episodes,
        "crash_rate": crash_count / eval_episodes,
        "final_distance_mean": float(np.mean(final_distances)),
        "final_distance_std": float(np.std(final_distances)),

        # Action metrics
        "action_mean": float(np.mean(action_arr)),
        "action_std": float(np.std(action_arr)),
        "action_distribution": {
            str(a): float(np.mean(action_arr == a))
            for a in range(4)
        },

        # Component returns
        "component_returns": component_means,

        # PPO config
        "ppo_config": {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
        },
    }

    # Save result
    with open(work_dir / f"{run_name}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
