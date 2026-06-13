from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CandidateRecord:
    generation: int
    candidate_id: str
    parent_ids: list[str]

    schema_version: str
    env_alias: str
    status: str
    validation_errors: list[str]

    repair_attempts: int
    repair_success: bool
    validation_errors_before_repair: list[str]
    validation_errors_after_repair: list[str]

    reflection_summary: str
    reward_code: str
    llm_rationale: str

    train_mean_return: float
    hidden_eval_return: float
    selection_score: float

    judge_score: float
    judge_reason: str
    judge_details: dict[str, Any]
    video_path: str

    # Backward-compatible warning fields.
    identity_warning_count: int = 0
    identity_warnings: dict[str, Any] = field(default_factory=dict)
    semantic_term_warning_count: int = 0
    semantic_term_warnings: dict[str, Any] = field(default_factory=dict)
    semantic_warning_count: int = 0
    semantic_warnings: dict[str, Any] = field(default_factory=dict)

    # New EG-RSA fields.
    prompt_paths: dict[str, Any] = field(default_factory=dict)
    prompt_budgets: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    lesson_ids: list[str] = field(default_factory=list)


class JsonlMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CandidateRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def top_candidates(
        self,
        k: int,
        schema_version: Optional[str] = None,
        env_alias: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        rows = self.load_all()

        if schema_version is not None:
            rows = [r for r in rows if r.get("schema_version") == schema_version]
        if env_alias is not None:
            rows = [r for r in rows if r.get("env_alias") == env_alias]

        rows = [r for r in rows if r.get("status") == "ok"]
        rows.sort(key=lambda r: float(r.get("selection_score", -1e18)), reverse=True)
        return rows[:k]
