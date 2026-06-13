from __future__ import annotations

from pathlib import Path
from typing import Any


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
        "Final EG-RSA framework requires envs/<task>/task_description.txt and envs/<task>/step.py. "
        "It does not synthesize extra observation/action range tables."
    )


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def infer_clean_env_interface(env_id: str, env_alias: str) -> dict[str, Any]:
    """
    Final EG-RSA task input.

    The LLM receives exactly Eureka-style task context:
      1. task_description.txt
      2. step.py

    The framework may add memory, lessons, feedback, schema, and parent reward code
    during iteration. It does not add a separate Gym-space-derived observation/action
    range table.
    """
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
                "environment understanding generated from task files",
                "reward schema and search plan generated from task files",
                "parent reward code",
                "training feedback",
                "STM/MTM/LTM lessons retrieved from memory",
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
