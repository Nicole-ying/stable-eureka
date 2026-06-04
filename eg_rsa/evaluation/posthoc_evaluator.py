from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class PosthocEvaluator:
    """Evaluate a saved policy with the original environment reward.

    This evaluator is for reporting only. Its output must not be used for reward
    editing or reward selection during EG-RSA search.
    """

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}

    def evaluate_model_path(
        self,
        model_path: Path,
        n_episodes: int = 5,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        env = gym.make(self.env_id, **self.env_kwargs)
        model = PPO.load(str(model_path), env=env)
        return self.evaluate_model(model, env, n_episodes, seed, deterministic)

    def evaluate_model(
        self,
        model,
        env,
        n_episodes: int = 5,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        returns = []
        lengths = []
        for episode_id in range(n_episodes):
            reset_out = env.reset(seed=None if seed is None else seed + episode_id)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            ep_return = 0.0
            ep_len = 0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, _ = step_out
                    done = bool(done)
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                ep_len += 1
            returns.append(ep_return)
            lengths.append(ep_len)
        return {
            "note": "Post-hoc oracle reward evaluation only. Not used by EG-RSA reward search.",
            "env_id": self.env_id,
            "num_episodes": int(n_episodes),
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "return_std": float(np.std(returns)) if returns else 0.0,
            "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "episode_returns": [float(x) for x in returns],
            "episode_lengths": [int(x) for x in lengths],
        }

    @staticmethod
    def save(path: Path, result: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
