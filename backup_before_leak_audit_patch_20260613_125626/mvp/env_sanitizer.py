from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _clip_float_list(x, max_items: int = 32) -> list[float] | str:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size > max_items:
        return f"<omitted:{arr.size}_values>"
    out = []
    for v in arr.tolist():
        if np.isinf(v):
            out.append(float(v))
        else:
            out.append(round(float(v), 6))
    return out


def _space_to_public_dict(space: spaces.Space) -> dict[str, Any]:
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": _clip_float_list(space.low),
            "high": _clip_float_list(space.high),
        }

    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
        }

    if isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": np.asarray(space.nvec).astype(int).tolist(),
        }

    if isinstance(space, spaces.MultiBinary):
        return {
            "type": "MultiBinary",
            "n": int(space.n) if isinstance(space.n, int) else list(space.n),
        }

    return {
        "type": type(space).__name__,
        "repr": repr(space),
    }


def infer_clean_env_interface(env_id: str, env_alias: str) -> dict[str, Any]:
    """
    从真实 gym env 中抽取 clean interface。

    重要：
      1. 不返回真实 env_id；
      2. 不返回原始 env reward；
      3. 不返回源码、docstring、官方 reward 描述；
      4. 不返回 observation/action 的人工语义解释。
    """
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        interface = {
            "env_alias": env_alias,
            "observation_space": _space_to_public_dict(obs_space),
            "action_space": _space_to_public_dict(act_space),
            "api": {
                "reset": "obs, info = env.reset()",
                "step": "next_obs, hidden_env_reward, terminated, truncated, info = env.step(action)",
                "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
                "reward_return": "total_reward, components",
            },
            "visibility_policy": {
                "hidden_from_reward_designer": [
                    "real env_id",
                    "raw environment source code",
                    "docstrings and comments",
                    "original env_reward",
                    "official reward formula",
                    "fitness_score implementation",
                    "benchmark id",
                ],
                "visible_to_reward_designer": [
                    "anonymous env alias",
                    "observation space shape/range",
                    "action space type/shape",
                    "done flag",
                    "info dictionary keys only if observed at runtime",
                    "closed-loop training statistics",
                ],
            },
        }
        return interface
    finally:
        env.close()
