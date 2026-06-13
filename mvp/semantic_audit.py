from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# Semantic warning audit
# ============================================================
#
# 设计原则：
#   1. 这里只做 warning，不做 hard fail；
#   2. 不直接影响训练、选择、validator；
#   3. 用于统计 LLM 是否在匿名接口下仍然推断 benchmark 身份
#      或给 obs 维度赋予物理语义；
#   4. 不写某个环境的专用逻辑，词表覆盖常见控制任务。
#
# 两类 warning：
#   A. identity warnings:
#      疑似 benchmark / agent / object identity 推断。
#
#   B. physical semantic warnings:
#      疑似 observation dimension physical semantics 推断。
#
# 注意：
#   - "goal" 太通用，已移除，避免 clean_plan 里 Task goal 字段误报；
#   - "progress/stability/effort/terminal" 是 schema 词，不在这里统计。
# ============================================================


IDENTITY_WARNING_TERMS = (
    # classic control / gym-like benchmark identity
    "cart",
    "pole",
    "cartpole",
    "lander",
    "lunar",
    "landing",
    "mountain",
    "mountaincar",
    "pendulum",
    "acrobot",

    # locomotion / robot-like identity
    "walker",
    "bipedal",
    "hopper",
    "cheetah",
    "ant",
    "humanoid",
    "robot",

    # task object identity
    "leg",
    "legs",
    "contact",
    "contacts",
    "thruster",
    "engine",
    "engines",
)


PHYSICAL_SEMANTIC_WARNING_TERMS = (
    # coordinate / state semantics
    "position",
    "positions",
    "velocity",
    "velocities",
    "angle",
    "angles",
    "angular",
    "coordinate",
    "coordinates",
    "x coordinate",
    "y coordinate",
    "height",
    "altitude",
    "distance",
    "origin",
    "target",

    # dynamics / behavior semantics
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
    "torque",
    "speed",
    "acceleration",
)


def _count_terms(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    lower = text.lower()
    counts: dict[str, int] = {}

    for term in terms:
        term_lower = term.lower()

        if " " in term_lower:
            n = lower.count(term_lower)
        else:
            # Use word boundary to reduce false positives.
            n = len(re.findall(rf"\b{re.escape(term_lower)}\b", lower))

        if n:
            counts[term] = n

    return counts


def _audit_term_group(bundle: dict[str, Any], terms: tuple[str, ...]) -> tuple[int, dict[str, dict[str, int]]]:
    per_artifact: dict[str, dict[str, int]] = {}
    total = 0

    for name, value in bundle.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        counts = _count_terms(text, terms)
        per_artifact[name] = counts
        total += sum(counts.values())

    return total, per_artifact


def audit_semantic_text_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Return semantic warning statistics.

    Backward-compatible fields:
      - semantic_warning_count
      - semantic_warnings

    New split fields:
      - identity_warning_count / identity_warnings
      - semantic_term_warning_count / semantic_term_warnings
    """
    identity_total, identity_warnings = _audit_term_group(bundle, IDENTITY_WARNING_TERMS)
    semantic_total, semantic_warnings = _audit_term_group(bundle, PHYSICAL_SEMANTIC_WARNING_TERMS)

    merged: dict[str, dict[str, int]] = {}
    for name in set(identity_warnings) | set(semantic_warnings):
        merged[name] = {}
        merged[name].update(identity_warnings.get(name, {}))
        merged[name].update(semantic_warnings.get(name, {}))

    return {
        "identity_warning_count": identity_total,
        "identity_warnings": identity_warnings,
        "semantic_term_warning_count": semantic_total,
        "semantic_term_warnings": semantic_warnings,

        # Backward-compatible total fields.
        "semantic_warning_count": identity_total + semantic_total,
        "semantic_warnings": merged,
    }


def save_semantic_audit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
