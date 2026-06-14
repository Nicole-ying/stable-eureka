"""
Masked LunarLander environment.
Hides the official reward function — the agent ONLY sees the generated reward.
The official reward is stored in info['_hidden_reward'] for evaluation ONLY.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Callable

RewardFn = Callable[[np.ndarray, int, np.ndarray, bool, dict], tuple[float, dict]]

# Observation labels (for reference, available to reward designers)
OBS_LABELS = [
    "x_pos",            # obs[0]: horizontal position, range ~[-2.5, 2.5]
    "y_pos",            # obs[1]: vertical position, range ~[-2.5, 2.5]
    "x_vel",            # obs[2]: horizontal velocity, range ~[-10, 10]
    "y_vel",            # obs[3]: vertical velocity, range ~[-10, 10]
    "angle",            # obs[4]: lander angle in radians, range ~[-π, π]
    "ang_vel",          # obs[5]: angular velocity, range ~[-10, 10]
    "left_leg_contact", # obs[6]: 0 or 1
    "right_leg_contact",# obs[7]: 0 or 1
]

# Action space
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine (downward thrust)
# 3: fire right orientation engine

# Landing pad is at (x=0, y=0). Goal: land between flags at x∈[-0.2, 0.2]


class MaskedLunarLander(gym.Wrapper):
    """
    Wraps LunarLander-v3, intercepts step() to:
    - Replace the official reward with a custom reward function
    - Store the official reward as info['_hidden_reward'] (for eval only)
    - Store reward components as info['_components']
    """

    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs: np.ndarray | None = None
        self._total_hidden_reward: float = 0.0
        self._total_generated_reward: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = np.array(obs, copy=True)
        self._total_hidden_reward = 0.0
        self._total_generated_reward = 0.0
        return obs, info

    def step(self, action):
        next_obs, hidden_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        # Call the custom reward function
        result = self.reward_fn(
            np.array(self._prev_obs, copy=True),
            int(action),
            np.array(next_obs, copy=True),
            done,
            dict(info),
        )

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("reward_fn must return (reward: float, components: dict)")

        generated_reward, components = result
        generated_reward = float(generated_reward)

        if not np.isfinite(generated_reward):
            raise ValueError(f"Generated reward is not finite: {generated_reward}")

        if not isinstance(components, dict):
            raise ValueError("Components must be a dict")

        # Store hidden reward and components (NEVER expose hidden_reward to reward_fn)
        safe_info = dict(info)
        safe_info["_hidden_reward"] = float(hidden_reward)
        safe_info["_generated_reward"] = generated_reward
        safe_info["_components"] = {str(k): float(v) for k, v in components.items()}

        self._total_hidden_reward += float(hidden_reward)
        self._total_generated_reward += generated_reward
        self._prev_obs = np.array(next_obs, copy=True)

        return next_obs, generated_reward, terminated, truncated, safe_info


def compile_reward_function(code: str) -> RewardFn:
    """Compile a reward function from source code string."""
    import ast
    import types

    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            raise ValueError("from-import statements not allowed in reward code")
        if isinstance(node, ast.Import):
            for alias in node.names:
                allowed = {"numpy"}
                name = alias.name.split(".")[0]
                if name not in allowed:
                    raise ValueError(
                        f"only 'import numpy' is allowed, got: import {alias.name}"
                    )

    module = types.ModuleType("reward_module")
    module.__dict__["np"] = np
    exec(compile(tree, filename="<reward_code>", mode="exec"), module.__dict__)

    if "compute_reward" not in module.__dict__:
        raise ValueError("reward code must define compute_reward(obs, action, next_obs, done, info)")
    return module.__dict__["compute_reward"]


def make_env(reward_fn: RewardFn) -> gym.Env:
    """Create a masked LunarLander-v3 environment."""
    env = gym.make("LunarLander-v3")
    return MaskedLunarLander(env, reward_fn)
