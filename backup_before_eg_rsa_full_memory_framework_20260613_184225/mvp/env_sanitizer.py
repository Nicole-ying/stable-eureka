from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _finite_ratio(x) -> float:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def _range_category_box(space: spaces.Box) -> str:
    low_finite = _finite_ratio(space.low)
    high_finite = _finite_ratio(space.high)

    if low_finite == 1.0 and high_finite == 1.0:
        return "fully_bounded"
    if low_finite == 0.0 and high_finite == 0.0:
        return "unbounded"
    return "partially_bounded"


def _space_to_public_dict(space: spaces.Space) -> dict[str, Any]:
    """
    Return an anonymized public space description.

    关键原则：
      - 不暴露具体 low/high 数值，避免 LLM 通过 benchmark bounds 反推环境；
      - 不暴露 observation 维度语义；
      - 只暴露实现 reward code 必需的 shape/type/cardinality。
    """
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "range_category": _range_category_box(space),
            "value_view": "normalized_anonymous_values",
            "normalization": "finite Box dimensions are scaled approximately into [-1, 1]; non-finite dimensions are clipped by a generic scale.",
            "dimension_semantics": "not_available",
        }

    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
            "dimension_semantics": "not_available",
        }

    if isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "shape": list(np.asarray(space.nvec).shape),
            "cardinality_view": "available_without_semantic_labels",
            "dimension_semantics": "not_available",
        }

    if isinstance(space, spaces.MultiBinary):
        return {
            "type": "MultiBinary",
            "shape": list(np.asarray(space.n).shape) if not isinstance(space.n, int) else [int(space.n)],
            "dimension_semantics": "not_available",
        }

    return {
        "type": type(space).__name__,
        "description": "custom_space_anonymized",
        "dimension_semantics": "not_available",
    }


def infer_clean_env_interface(env_id: str, env_alias: str) -> dict[str, Any]:
    """
    抽取只允许进入 LLM 的 clean interface。

    通用匿名化原则：
      - 真实环境标识只在 private runtime 中使用；
      - 原始环境返回的私有评估信号不出现在这里；
      - 源码、docstring、官方奖励说明、fitness 实现都不出现在这里；
      - observation/action 不提供物理语义；
      - observation 不提供具体 low/high benchmark bounds；
      - reward function 看到的是 normalized anonymous observations。
    """
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space

        return {
            "env_alias": env_alias,
            "observation_space": _space_to_public_dict(obs_space),
            "action_space": _space_to_public_dict(act_space),
            "reward_observation_view": {
                "obs": "normalized anonymous observation with the same shape as the original observation",
                "next_obs": "normalized anonymous next observation with the same shape as the original observation",
                "raw_observation_semantics": "not_available",
            },
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
