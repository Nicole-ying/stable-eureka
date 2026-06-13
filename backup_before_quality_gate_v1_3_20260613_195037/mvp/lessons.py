from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_lesson(
    lesson: dict[str, Any],
    *,
    scope: str,
    env_alias: str,
    generation: int | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    out = dict(lesson)
    out.setdefault("lesson_id", f"{scope}_{uuid.uuid4().hex[:10]}")
    out.setdefault("scope", scope)
    out.setdefault("lesson_type", "general")
    out.setdefault("condition", "")
    out.setdefault("observation", "")
    out.setdefault("explanation", "")
    out.setdefault("recommendation", "")
    out.setdefault("confidence", 0.5)
    out.setdefault("reuse_policy", "same_env" if scope != "cross_environment" else "global")
    out.setdefault("env_alias", env_alias)
    if generation is not None:
        out.setdefault("generation", generation)
    if candidate_id is not None:
        out.setdefault("candidate_id", candidate_id)
    return out


def compact_lesson_line(row: dict[str, Any]) -> str:
    lid = row.get("lesson_id", "lesson")
    ltype = row.get("lesson_type", "general")
    cond = str(row.get("condition", "")).strip()
    rec = str(row.get("recommendation", "")).strip()
    conf = row.get("confidence", "")
    return f"- [{lid}] type={ltype}, confidence={conf}, condition={cond}, recommendation={rec}"


def retrieve_memory_context(
    *,
    stm_top: list[dict[str, Any]],
    candidate_lessons_path: Path,
    env_lessons_path: Path,
    ltm_lessons_path: Path,
    env_alias: str,
    candidate_lesson_top_k: int,
    env_lesson_top_k: int,
    ltm_lesson_top_k: int,
    max_chars: int,
) -> str:
    candidate_lessons = read_jsonl(candidate_lessons_path)
    env_lessons = read_jsonl(env_lessons_path)
    ltm_lessons = read_jsonl(ltm_lessons_path)

    parent_ids = {r.get("candidate_id") for r in stm_top}
    candidate_lessons = [
        x for x in candidate_lessons
        if x.get("candidate_id") in parent_ids or x.get("env_alias") == env_alias
    ][-candidate_lesson_top_k:]

    env_lessons = [
        x for x in env_lessons
        if x.get("env_alias") == env_alias
    ][-env_lesson_top_k:]

    # Do not retrieve current-environment lessons from LTM. Otherwise the
    # same generation's environment lessons can immediately re-enter as
    # "cross-environment" memory and pollute the next prompt.
    ltm_lessons = [
        x for x in ltm_lessons
        if x.get("reuse_policy") in ("global", "similar_env", None)
        and x.get("env_alias") != env_alias
    ][-ltm_lesson_top_k:]

    parts = []
    parts.append("Relevant candidate-level lessons:")
    parts.extend(compact_lesson_line(x) for x in candidate_lessons)
    if not candidate_lessons:
        parts.append("- none")

    parts.append("")
    parts.append("Relevant environment-level lessons:")
    parts.extend(compact_lesson_line(x) for x in env_lessons)
    if not env_lessons:
        parts.append("- none")

    parts.append("")
    parts.append("Relevant cross-environment lessons:")
    parts.extend(compact_lesson_line(x) for x in ltm_lessons)
    if not ltm_lessons:
        parts.append("- none")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def pack_generation_evidence(
    *,
    generation: int,
    records: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    gen_rows = [r for r in records if int(r.get("generation", -1)) == generation]
    ok_rows = [r for r in gen_rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: float(r.get("selection_score", -1e18)), reverse=True)

    def slim(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": r.get("candidate_id"),
            "parent_ids": r.get("parent_ids", []),
            "status": r.get("status"),
            "selection_score": r.get("selection_score"),
            "private_eval_return": r.get("hidden_eval_return"),
            "generated_return": r.get("train_mean_return"),
            "generated_minus_private": float(r.get("train_mean_return", 0.0)) - float(r.get("hidden_eval_return", 0.0)),
            "repair_attempts": r.get("repair_attempts", 0),
            "repair_success": r.get("repair_success", False),
            "validation_errors": r.get("validation_errors", []),
            "diagnostics": r.get("diagnostics", {}),
            "reward_code_head": str(r.get("reward_code", ""))[:2500],
            "llm_rationale": str(r.get("llm_rationale", ""))[:1000],
        }

    return {
        "generation": generation,
        "num_candidates": len(gen_rows),
        "num_ok": len(ok_rows),
        "top_candidates": [slim(r) for r in ok_rows[:top_k]],
        "bottom_candidates": [slim(r) for r in ok_rows[-top_k:]],
        "failed_candidates": [slim(r) for r in gen_rows if r.get("status") != "ok"],
    }


def pack_candidate_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Compact evidence for candidate-level lesson extraction."""
    return {
        "generation": record.get("generation"),
        "candidate_id": record.get("candidate_id"),
        "parent_ids": record.get("parent_ids", []),
        "status": record.get("status"),
        "selection_score": record.get("selection_score"),
        "private_eval_return": record.get("hidden_eval_return"),
        "generated_return": record.get("train_mean_return"),
        "generated_minus_private": float(record.get("train_mean_return", 0.0)) - float(record.get("hidden_eval_return", 0.0)),
        "repair_attempts": record.get("repair_attempts", 0),
        "repair_success": record.get("repair_success", False),
        "validation_errors": record.get("validation_errors", []),
        "diagnostics": record.get("diagnostics", {}),
        "reward_code_head": str(record.get("reward_code", ""))[:3500],
        "llm_rationale": str(record.get("llm_rationale", ""))[:1500],
    }
