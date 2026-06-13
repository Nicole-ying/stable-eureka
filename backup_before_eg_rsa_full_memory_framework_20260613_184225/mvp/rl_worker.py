from __future__ import annotations

import ast
import math
import types
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from .config import RLConfig


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
    """
    Compile LLM reward code into compute_reward(obs, action, next_obs, done, info).

    关键边界：
      - 不允许 env_reward 参数；
      - 不允许 import；
      - 不允许读取 hidden evaluator / benchmark / fitness；
      - 只能使用 np、math、python builtins。
    """
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

    fn = module.__dict__["compute_reward"]
    return fn


class ObservationAnonymizer:
    """
    Generic observation anonymizer for reward computation.

    策略：
      - policy 仍然看 raw obs；
      - reward function 只看 normalized anonymous obs；
      - 对 Box 空间：有限 bounds 用中心/半径归一化到约 [-1,1]；
        非有限维度用通用尺度 tanh 压缩；
      - 对非 Box 空间：转成 float array，做通用尺度压缩；
      - 不包含任何环境专属逻辑。
    """

    def __init__(self, observation_space: spaces.Space):
        self.observation_space = observation_space

        if isinstance(observation_space, spaces.Box):
            low = np.asarray(observation_space.low, dtype=np.float32)
            high = np.asarray(observation_space.high, dtype=np.float32)
            finite = np.isfinite(low) & np.isfinite(high) & (high > low)

            center = np.zeros_like(low, dtype=np.float32)
            scale = np.ones_like(low, dtype=np.float32)

            center[finite] = (low[finite] + high[finite]) / 2.0
            scale[finite] = (high[finite] - low[finite]) / 2.0
            scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)

            self.box_finite = finite
            self.center = center
            self.scale = scale
        else:
            self.box_finite = None
            self.center = None
            self.scale = None

    def transform(self, obs):
        arr = np.asarray(obs, dtype=np.float32)

        if isinstance(self.observation_space, spaces.Box):
            out = np.zeros_like(arr, dtype=np.float32)

            finite = self.box_finite
            out[finite] = (arr[finite] - self.center[finite]) / self.scale[finite]

            # 非有限 bound 维度不暴露原始尺度，用 generic tanh 压缩。
            nonfinite = ~finite
            out[nonfinite] = np.tanh(arr[nonfinite] / 5.0)

            out = np.clip(out, -5.0, 5.0)
            return out.astype(np.float32)

        return np.tanh(arr / 5.0).astype(np.float32)


class RewardFunctionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs = None
        self._prev_obs_reward_view = None
        self._anonymizer = ObservationAnonymizer(env.observation_space)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        self._prev_obs_reward_view = self._anonymizer.transform(obs)
        return obs, info

    def step(self, action):
        next_obs, hidden_env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        next_obs_reward_view = self._anonymizer.transform(next_obs)

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

        self._prev_obs = next_obs
        self._prev_obs_reward_view = next_obs_reward_view
        return next_obs, reward, terminated, truncated, safe_info


class RLWorker:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg

    def train_and_eval(self, reward_code: str, ckpt_path: Path) -> dict[str, float]:
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
        train_env.close()

        eval_env = RewardFunctionWrapper(gym.make(self.cfg.env_id), reward_fn)
        generated_returns = []
        hidden_returns = []
        episode_lengths = []

        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            gen_ret = 0.0
            hid_ret = 0.0
            ep_len = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)
                gen_ret += float(reward)
                hid_ret += float(info.get("_hidden_env_reward", 0.0))
                ep_len += 1

            generated_returns.append(gen_ret)
            hidden_returns.append(hid_ret)
            episode_lengths.append(ep_len)

        eval_env.close()

        return {
            "eval_generated_return": float(np.mean(generated_returns)),
            "eval_hidden_return": float(np.mean(hidden_returns)),
            "eval_episode_length": float(np.mean(episode_lengths)),
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
