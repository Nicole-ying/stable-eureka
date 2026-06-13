from __future__ import annotations

import hashlib
from dataclasses import dataclass


# ============================================================
# PublicTaskSpec
# ============================================================
#
# 只允许进入 LLM prompt 的公开任务信息。
#
# 设计原则：
#   1. 不写真实 env_id 的语义解释；
#   2. 不写 observation 每一维的物理含义；
#   3. 不写 action 每个编号/维度的真实含义；
#   4. 不写官方 reward decomposition；
#   5. 不写 benchmark / fitness / hidden reward 相关内容。
#
# 这样做的目的：
#   避免 LLM 依靠已知 Gym 环境知识或官方奖励模板，
#   而是基于 clean interface + 训练反馈搜索 reward。
# ============================================================


@dataclass(frozen=True)
class PublicTaskSpec:
    task_goal: str
    task_style: str
    forbidden_terms: tuple[str, ...]


@dataclass(frozen=True)
class PrivateTaskSpec:
    env_id: str
    hidden_eval_source: str = "env_reward"
    benchmark_id: str | None = None


DEFAULT_FORBIDDEN_TERMS = (
    "env_reward",
    "fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "compute_fitness_score",
    "LunarLander",
    "BipedalWalker",
    "CartPole",
    "gymnasium.envs",
)


PUBLIC_TASK_SPECS: dict[str, PublicTaskSpec] = {
    "LunarLander-v3": PublicTaskSpec(
        task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
        task_style=(
            "Prefer dense, bounded shaping terms that encourage progress, stability, "
            "low unnecessary control effort, and appropriate terminal behavior."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
    "LunarLander-v2": PublicTaskSpec(
        task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
        task_style=(
            "Prefer dense, bounded shaping terms that encourage progress, stability, "
            "low unnecessary control effort, and appropriate terminal behavior."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
    "BipedalWalker-v3": PublicTaskSpec(
        task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
        task_style=(
            "Prefer dense, bounded shaping terms that encourage progress, stability, "
            "smooth transitions, low unnecessary control effort, and robust completion."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
    "CartPole-v1": PublicTaskSpec(
        task_goal="Control the agent to maintain task success for as long as possible.",
        task_style=(
            "Prefer bounded shaping terms that encourage stable observations, smooth transitions, "
            "and appropriate terminal penalties."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
}


PRIVATE_TASK_SPECS: dict[str, PrivateTaskSpec] = {
    "LunarLander-v3": PrivateTaskSpec(env_id="LunarLander-v3", benchmark_id="LunarLander-v3"),
    "LunarLander-v2": PrivateTaskSpec(env_id="LunarLander-v2", benchmark_id="LunarLander-v2"),
    "BipedalWalker-v3": PrivateTaskSpec(env_id="BipedalWalker-v3", benchmark_id="BipedalWalker-v3"),
    "CartPole-v1": PrivateTaskSpec(env_id="CartPole-v1", benchmark_id="CartPole-v1"),
}


def make_env_alias(env_id: str) -> str:
    digest = hashlib.sha1(env_id.encode("utf-8")).hexdigest()[:8]
    return f"Env-{digest}"


def get_public_task_spec(env_id: str) -> PublicTaskSpec:
    return PUBLIC_TASK_SPECS.get(
        env_id,
        PublicTaskSpec(
            task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
            task_style=(
                "Use clean observation/action interface information and closed-loop feedback. "
                "Do not assume any official reward template."
            ),
            forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
        ),
    )


def get_private_task_spec(env_id: str) -> PrivateTaskSpec:
    return PRIVATE_TASK_SPECS.get(env_id, PrivateTaskSpec(env_id=env_id, benchmark_id=env_id))
