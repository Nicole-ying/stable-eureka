from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# Generic semantic warning terms
# ============================================================
#
# 注意：
#   1. 这些词不是 hard leak terms，不直接导致实验失败；
#   2. 它们用于统计 LLM 是否在匿名接口下仍然给 obs 维度赋予物理语义；
#   3. 不只针对 CartPole/LunarLander，而是覆盖通用控制任务中常见的
#      物理/benchmark/目标语义词。
# ============================================================


SEMANTIC_WARNING_TERMS = (
    # benchmark / object identity style terms
    "cart",
    "pole",
    "lander",
    "landing",
    "leg",
    "contact",
    "mountain",
    "car",
    "pendulum",
    "acrobot",
    "walker",
    "bipedal",
    "hopper",
    "cheetah",
    "ant",
    "humanoid",
    "robot",

    # physical coordinate semantics
    "position",
    "velocity",
    "angle",
    "angular",
    "coordinate",
    "x coordinate",
    "y coordinate",
    "height",
    "altitude",
    "distance",
    "origin",
    "target",
    "goal",

    # dynamics / task outcome semantics
    "upright",
    "balance",
    "balancing",
    "fall",
    "falling",
    "crash",
    "land",
    "landed",
    "success flag",
    "failure",
    "thruster",
    "engine",
    "torque",
    "speed",
    "acceleration",
)


def _count_terms(text: str) -> dict[str, int]:
    lower = text.lower()
    counts: dict[str, int] = {}

    for term in SEMANTIC_WARNING_TERMS:
        # word-ish boundary for normal words, substring fallback for multi-word phrases
        if " " in term:
            n = lower.count(term.lower())
        else:
            n = len(re.findall(rf"\b{re.escape(term.lower())}\b", lower))
        if n:
            counts[term] = n

    return counts


def audit_semantic_text_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    per_artifact = {}
    total = 0

    for name, value in bundle.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        counts = _count_terms(text)
        per_artifact[name] = counts
        total += sum(counts.values())

    return {
        "semantic_warning_count": total,
        "semantic_warnings": per_artifact,
    }


def save_semantic_audit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
