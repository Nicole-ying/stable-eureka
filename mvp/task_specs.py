from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivateTaskSpec:
    """
    Runtime-only task identity.

    EG-RSA follows Eureka-style task input:
      - task_description.txt
      - step.py

    The real env_id is used only by runtime to create Gym environments
    and evaluate policies. It is not used to synthesize extra task
    semantics beyond the Eureka task files.
    """
    env_id: str
    hidden_eval_source: str = "env_reward"
    benchmark_id: str | None = None


PRIVATE_TASK_SPECS: dict[str, PrivateTaskSpec] = {
    "LunarLander-v3": PrivateTaskSpec(env_id="LunarLander-v3", benchmark_id="LunarLander-v3"),
    "LunarLander-v2": PrivateTaskSpec(env_id="LunarLander-v2", benchmark_id="LunarLander-v2"),
    "BipedalWalker-v3": PrivateTaskSpec(env_id="BipedalWalker-v3", benchmark_id="BipedalWalker-v3"),
    "CartPole-v1": PrivateTaskSpec(env_id="CartPole-v1", benchmark_id="CartPole-v1"),
}


def make_env_alias(env_id: str) -> str:
    digest = hashlib.sha1(env_id.encode("utf-8")).hexdigest()[:8]
    return f"Env-{digest}"


def get_private_task_spec(env_id: str) -> PrivateTaskSpec:
    return PRIVATE_TASK_SPECS.get(env_id, PrivateTaskSpec(env_id=env_id, benchmark_id=env_id))
