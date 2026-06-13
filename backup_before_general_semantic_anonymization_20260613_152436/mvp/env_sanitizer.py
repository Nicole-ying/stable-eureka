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
    抽取只允许进入 LLM 的 clean interface。

    注意：
      - 真实环境标识只在 private runtime 中使用；
      - 原始环境返回的私有评估信号不出现在这里；
      - 源码、docstring、官方奖励说明、fitness 实现都不出现在这里；
      - observation/action 只暴露形状、范围、类型，不暴露人工语义解释。
    """
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space

        return {
            "env_alias": env_alias,
            "observation_space": _space_to_public_dict(obs_space),
            "action_space": _space_to_public_dict(act_space),
            "api": {
                "reset": "obs, info = reset()",
                "step_visible_outputs": [
                    "next_obs",
                    "terminated",
                    "truncated",
                    "info",
                ],
                "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
                "reward_return": "total_reward, components",
            },
        }
    finally:
        env.close()
