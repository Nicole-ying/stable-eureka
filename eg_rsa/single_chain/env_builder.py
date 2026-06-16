from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Type

import gymnasium as gym
import numpy as np

from .reward_validation import indent_as_class_method
from .json_tools import write_text


class RewardInfoWrapper(gym.Wrapper):
    """Attach rollout diagnostics without changing environment physics."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._eg_episode_step = 0

    def reset(self, **kwargs):
        self._eg_episode_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._eg_episode_step += 1
        info = dict(info)
        info["_eg_episode_step"] = self._eg_episode_step
        info["_eg_action"] = _safe_action_to_json(action)
        info["_eg_done"] = bool(terminated or truncated)
        info.update(_terminal_outcome_flags(self.env, obs, terminated, truncated))
        if terminated or truncated:
            info["_eg_final_state"] = np.asarray(obs, dtype=float).tolist()
        return obs, reward, terminated, truncated, info


def _terminal_outcome_flags(env: gym.Env, obs: Any, terminated: bool, truncated: bool) -> Dict[str, float]:
    """Create task-generic terminal diagnostics, with LunarLander support.

    LunarLander's original success event is encoded as `not lander.awake`; the
    generated reward code cannot reliably infer that from state alone. These flags
    make evaluator success visible to trainer/evidence without modifying physics.
    """
    flags = {
        "success_like_terminal": 0.0,
        "safe_landing_terminal": 0.0,
        "unsafe_terminal": 0.0,
        "crash_terminal": 0.0,
        "out_of_bounds": 0.0,
        "out_of_bounds_terminal": 0.0,
        "timeout_terminal": 1.0 if truncated else 0.0,
    }
    if not (terminated or truncated):
        return flags

    arr = np.asarray(obs, dtype=float).reshape(-1)
    x = float(arr[0]) if arr.size >= 1 else 0.0
    unwrapped = getattr(env, "unwrapped", env)
    game_over = bool(getattr(unwrapped, "game_over", False))
    lander = getattr(unwrapped, "lander", None)
    lander_awake = bool(getattr(lander, "awake", True)) if lander is not None else True

    out_of_bounds = bool(abs(x) >= 1.0)
    safe_landing = bool(terminated and not truncated and (not lander_awake) and not game_over and not out_of_bounds)
    crash = bool(terminated and not truncated and game_over and not safe_landing and not out_of_bounds)

    flags["out_of_bounds"] = 1.0 if out_of_bounds else 0.0
    flags["out_of_bounds_terminal"] = 1.0 if out_of_bounds else 0.0
    flags["safe_landing_terminal"] = 1.0 if safe_landing else 0.0
    flags["success_like_terminal"] = 1.0 if safe_landing else 0.0
    flags["crash_terminal"] = 1.0 if crash else 0.0
    flags["unsafe_terminal"] = 1.0 if crash else 0.0
    return flags


def _safe_action_to_json(action: Any) -> Any:
    arr = np.asarray(action)
    if arr.shape == ():
        value = arr.item()
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value)
        return str(value)
    return arr.astype(float).tolist()


def prepare_env_code(base_env_code_dir: str | Path, output_env_code_dir: str | Path, reward_code: str) -> Path:
    """Copy Eureka env_code and append the generated compute_reward method."""
    base_env_code_dir = Path(base_env_code_dir)
    output_env_code_dir = Path(output_env_code_dir)
    if output_env_code_dir.exists():
        shutil.rmtree(output_env_code_dir)
    shutil.copytree(base_env_code_dir, output_env_code_dir)

    env_py = output_env_code_dir / "env.py"
    if not env_py.exists():
        raise FileNotFoundError(f"env.py not found in {output_env_code_dir}")

    original = env_py.read_text(encoding="utf-8")
    patched = original.rstrip() + "\n" + indent_as_class_method(reward_code)
    write_text(env_py, patched)
    return env_py


def load_env_class(env_py: str | Path, class_name: str) -> Type[gym.Env]:
    env_py = Path(env_py)
    module_name = f"eg_rsa_generated_env_{abs(hash(str(env_py)))}"
    spec = importlib.util.spec_from_file_location(module_name, env_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {env_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    env_cls = getattr(module, class_name)
    return env_cls


def make_single_env(env_cls: Type[gym.Env], env_kwargs: Dict[str, Any], max_episode_steps: int | None, seed: int | None):
    env = env_cls(**(env_kwargs or {}))
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = RewardInfoWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
