from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ENV_ID_TO_EUREKA_DIRS: dict[str, list[str]] = {
    "LunarLander-v3": ["lunar_lander"],
    "LunarLander-v2": ["lunar_lander"],
    "BipedalWalker-v3": ["bipedal_walker"],
    "CartPole-v1": ["cartpole", "cart_pole"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_env_dirs(env_id: str) -> list[Path]:
    root = _repo_root()
    out: list[Path] = []

    for name in ENV_ID_TO_EUREKA_DIRS.get(env_id, []):
        out.append(root / "envs" / name)

    stem = env_id.split("-")[0]
    candidates = {
        stem,
        stem.lower(),
        stem.replace("LunarLander", "lunar_lander"),
        stem.replace("BipedalWalker", "bipedal_walker"),
        stem.replace("CartPole", "cartpole"),
    }
    for c in candidates:
        out.append(root / "envs" / c)

    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _find_eureka_file(env_id: str, filename: str) -> Path:
    for d in _candidate_env_dirs(env_id):
        p = d / filename
        if p.exists():
            return p

    searched = "\n".join(str(d / filename) for d in _candidate_env_dirs(env_id))
    raise FileNotFoundError(
        f"Eureka-style file not found for env_id={env_id}: {filename}\n"
        f"Searched:\n{searched}\n\n"
        "EG-RSA eureka_clean requires envs/<task>/task_description.txt and envs/<task>/step.py. "
        "The framework will not synthesize extra task-interface fields silently."
    )


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _anonymous_space(space: spaces.Space) -> dict[str, Any]:
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "dimension_semantics": "not_available",
        }
    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
            "dimension_semantics": "not_available",
        }
    return {
        "type": type(space).__name__,
        "dimension_semantics": "not_available",
    }


class ObservationAnonymizer:
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
            out[~finite] = np.tanh(arr[~finite] / 5.0)
            return np.clip(out, -5.0, 5.0).astype(np.float32)

        return np.tanh(arr / 5.0).astype(np.float32)


def infer_clean_env_interface(
    env_id: str,
    env_alias: str,
    interface_mode: str = "eureka_clean",
) -> dict[str, Any]:
    if interface_mode == "anonymous_clean":
        env = gym.make(env_id)
        try:
            return {
                "interface_mode": "anonymous_clean",
                "env_alias": env_alias,
                "observation_space": _anonymous_space(env.observation_space),
                "action_space": _anonymous_space(env.action_space),
                "reward_function_contract": {
                    "signature": "compute_reward(obs, action, next_obs, done, info)",
                    "visible_inputs": ["obs", "action", "next_obs", "done", "info"],
                    "return": "float(total_reward), components_dict",
                },
            }
        finally:
            env.close()

    if interface_mode != "eureka_clean":
        raise ValueError(f"Unsupported interface_mode={interface_mode}. Use eureka_clean or anonymous_clean.")

    task_path = _find_eureka_file(env_id, "task_description.txt")
    step_path = _find_eureka_file(env_id, "step.py")

    return {
        "interface_mode": "eureka_clean",
        "env_alias": env_alias,
        "eureka_task_description": _read_file(task_path),
        "eureka_step_code": _read_file(step_path),
        "source_files": {
            "task_description": str(task_path),
            "step": str(step_path),
        },
        "reward_function_contract": {
            "signature": "compute_reward(obs, action, next_obs, done, info)",
            "visible_inputs": ["obs", "action", "next_obs", "done", "info"],
            "return": "float(total_reward), components_dict",
        },
        "input_boundary": {
            "allowed": [
                "task_description.txt",
                "step.py",
                "LLM reasoning over observation/action semantics",
                "parent reward code",
                "training feedback",
                "lessons retrieved from memory",
            ],
            "generated_code_forbidden": [
                "env_reward",
                "official reward formula",
                "fitness_score",
                "compute_fitness_score",
                "hidden evaluator implementation",
                "expert reward template",
            ],
        },
    }
