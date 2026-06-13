from __future__ import annotations

import ast
import math
import types
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from .config import RLConfig
from .env_sanitizer import ObservationAnonymizer


RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, bool, dict], tuple[float, dict]]


FORBIDDEN_NAMES = {
    "env_reward",
    "fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
    "_hidden_env_reward",
    "compute_fitness_score",
}


def compile_reward_function(reward_code: str) -> RewardFn:
    raw_lower = reward_code.lower()
    for name in FORBIDDEN_NAMES:
        if name.lower() in raw_lower:
            raise ValueError(f"forbidden token appears in reward code: {name}")

    tree = ast.parse(reward_code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.ClassDef, ast.Lambda)):
            raise ValueError(f"unsupported syntax in reward code: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden name in reward code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden attribute in reward code: {node.attr}")

    module = types.ModuleType("reward_module")
    module.__dict__["np"] = np
    module.__dict__["math"] = math
    exec(compile(tree, filename="<reward_code>", mode="exec"), module.__dict__)

    if "compute_reward" not in module.__dict__:
        raise ValueError("reward code must define compute_reward(obs, action, next_obs, done, info)")
    return module.__dict__["compute_reward"]


class RewardFunctionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: RewardFn, interface_mode: str = "eureka_clean"):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.interface_mode = interface_mode
        self._prev_obs_reward_view = None
        self._anonymizer = ObservationAnonymizer(env.observation_space)

    def _reward_view(self, obs):
        if self.interface_mode == "anonymous_clean":
            return self._anonymizer.transform(obs)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs_reward_view = self._reward_view(obs)
        return obs, info

    def step(self, action):
        next_obs, hidden_env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        next_obs_reward_view = self._reward_view(next_obs)

        out = self.reward_fn(
            self._prev_obs_reward_view,
            action,
            next_obs_reward_view,
            done,
            info,
        )
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError("compute_reward must return (reward, components)")

        reward, components = out
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("generated reward is not finite")
        if not isinstance(components, dict):
            raise ValueError("reward components must be a dict")

        safe_info = dict(info)
        safe_info["_hidden_env_reward"] = float(hidden_env_reward)
        safe_info["_reward_components"] = {str(k): float(v) for k, v in components.items()}

        self._prev_obs_reward_view = next_obs_reward_view
        return next_obs, reward, terminated, truncated, safe_info


class RLWorker:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        self.interface_mode = getattr(cfg, "interface_mode", "eureka_clean")

    def train_and_eval(self, reward_code: str, ckpt_path: Path) -> dict[str, object]:
        reward_fn = compile_reward_function(reward_code)

        train_env = RewardFunctionWrapper(
            gym.make(self.cfg.env_id),
            reward_fn,
            interface_mode=self.interface_mode,
        )
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=self.cfg.learning_rate,
            gamma=self.cfg.gamma,
        )
        model.learn(total_timesteps=self.cfg.total_timesteps)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt_path))
        train_env.close()

        eval_env = RewardFunctionWrapper(
            gym.make(self.cfg.env_id),
            reward_fn,
            interface_mode=self.interface_mode,
        )

        generated_returns = []
        hidden_returns = []
        episode_lengths = []
        component_returns: dict[str, list[float]] = {}
        action_values = []

        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            gen_ret = 0.0
            hid_ret = 0.0
            ep_len = 0
            comp_sum: dict[str, float] = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_values.append(np.asarray(action, dtype=float).reshape(-1))
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)

                gen_ret += float(reward)
                hid_ret += float(info.get("_hidden_env_reward", 0.0))
                ep_len += 1

                for k, v in info.get("_reward_components", {}).items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + float(v)

            generated_returns.append(gen_ret)
            hidden_returns.append(hid_ret)
            episode_lengths.append(ep_len)

            for k, v in comp_sum.items():
                component_returns.setdefault(k, []).append(v)

        eval_env.close()

        action_arr = np.concatenate(action_values) if action_values else np.zeros((1,), dtype=float)
        component_mean = {k: float(np.mean(v)) for k, v in component_returns.items()}

        return {
            "eval_generated_return": float(np.mean(generated_returns)),
            "eval_hidden_return": float(np.mean(hidden_returns)),
            "eval_episode_length": float(np.mean(episode_lengths)),
            "component_returns": component_mean,
            "diagnostics": {
                "generated_private_gap": float(np.mean(generated_returns) - np.mean(hidden_returns)),
                "action_mean": float(np.mean(action_arr)),
                "action_std": float(np.std(action_arr)),
                "episode_length_mean": float(np.mean(episode_lengths)),
                "component_returns": component_mean,
            },
        }

    def render_rollout_video(self, ckpt_path: Path, video_path: Path) -> Path:
        model = PPO.load(str(ckpt_path))
        env = gym.make(self.cfg.env_id, render_mode="rgb_array")
        obs, _ = env.reset()
        done = False
        frames = []

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

        video_path.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            imageio.mimsave(video_path, frames, fps=30)
        env.close()
        return video_path
