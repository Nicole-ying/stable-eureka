from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from .config import RLConfig


RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float, bool, dict], float]


def compile_reward_function(reward_code: str) -> RewardFn:
    """Compile LLM reward code safely into a callable compute_reward function."""
    tree = ast.parse(reward_code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.ClassDef, ast.Lambda)):
            raise ValueError(f"unsupported syntax in reward code: {type(node).__name__}")
    module = types.ModuleType("reward_module")
    module.__dict__["np"] = np
    exec(compile(tree, filename="<reward_code>", mode="exec"), module.__dict__)
    if "compute_reward" not in module.__dict__:
        raise ValueError("reward code must define compute_reward(obs, action, next_obs, env_reward, done, info)")
    fn = module.__dict__["compute_reward"]
    return fn


class RewardFunctionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        reward = float(self.reward_fn(self._prev_obs, action, next_obs, env_reward, done, info))
        self._prev_obs = next_obs
        return next_obs, reward, terminated, truncated, info


class RLWorker:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg

    def train_and_eval(self, reward_code: str, ckpt_path: Path) -> float:
        reward_fn = compile_reward_function(reward_code)
        train_env = RewardFunctionWrapper(gym.make(self.cfg.env_id), reward_fn)
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

        eval_env = RewardFunctionWrapper(gym.make(self.cfg.env_id), reward_fn)
        returns = []
        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_ret += reward
            returns.append(ep_ret)
        return float(np.mean(returns))

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
            done = terminated or truncated
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)
        env.close()
        return video_path
