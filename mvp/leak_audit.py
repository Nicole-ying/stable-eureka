from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class LeakAuditError(RuntimeError):
    pass


# ============================================================
# Leak audit terms
# ============================================================
#
# 这些词只用于本地审计，不进入 RewardCoder prompt。
#
# 目标：
#   1. clean_interface / clean_plan / public_schema 不能出现真实环境名；
#   2. 不能出现原始奖励、隐藏评估、benchmark、fitness 等私有信号的显式变量名；
#   3. 不能出现旧 Gym 任务名，避免 LLM 用通用知识套官方 reward。
# ============================================================


DEFAULT_LEAK_TERMS = (
    "env_reward",
    "hidden_env_reward",
    "_hidden_env_reward",
    "fitness_score",
    "compute_fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
    "LunarLander",
    "BipedalWalker",
    "CartPole",
    "Acrobot",
    "MountainCar",
    "Pendulum",
)


def _normalize_terms(
    env_id: str | None = None,
    extra_terms: Iterable[str] | None = None,
) -> list[str]:
    terms = set(DEFAULT_LEAK_TERMS)

    if env_id:
        terms.add(env_id)
        terms.add(env_id.lower())
        terms.add(env_id.replace("-", ""))
        terms.add(env_id.replace("-", "_"))
        terms.add(env_id.split("-")[0])

    if extra_terms:
        for term in extra_terms:
            if term:
                terms.add(str(term))

    return sorted(t for t in terms if t)


def audit_text_bundle(
    bundle: dict[str, Any],
    env_id: str | None = None,
    extra_terms: Iterable[str] | None = None,
) -> dict[str, Any]:
    terms = _normalize_terms(env_id=env_id, extra_terms=extra_terms)
    violations = []

    for name, value in bundle.items():
        text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
        text_lower = text.lower()

        for term in terms:
            if term.lower() in text_lower:
                violations.append(
                    {
                        "artifact": name,
                        "term": term,
                    }
                )

    return {
        "ok": len(violations) == 0,
        "num_violations": len(violations),
        "violations": violations,
    }


def assert_no_leak_text(
    name: str,
    text: str,
    env_id: str | None = None,
    extra_terms: Iterable[str] | None = None,
) -> None:
    audit = audit_text_bundle({name: text}, env_id=env_id, extra_terms=extra_terms)
    if not audit["ok"]:
        raise LeakAuditError(f"Leak audit failed for {name}: {audit['violations']}")


def save_audit_report(audit: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
