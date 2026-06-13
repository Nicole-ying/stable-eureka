#!/usr/bin/env bash
set -euo pipefail

echo "[fix] patch mvp/memory.py for Python 3.8/3.9 compatibility..."

cp mvp/memory.py "mvp/memory.py.bak_py38_$(date +%Y%m%d_%H%M%S)"

cat > mvp/memory.py <<'PY'
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
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

        # 关键：旧泄露 memory 没有 schema_version/env_alias/status 字段，自动被过滤。
        if schema_version is not None:
            rows = [r for r in rows if r.get("schema_version") == schema_version]
        if env_alias is not None:
            rows = [r for r in rows if r.get("env_alias") == env_alias]

        rows = [r for r in rows if r.get("status") == "ok"]
        rows.sort(key=lambda r: float(r.get("selection_score", -1e18)), reverse=True)
        return rows[:k]
PY

python -m py_compile mvp/memory.py

echo "[fix] done."
echo ""
echo "Now retry:"
echo "python run_mvp.py --config mvp/configs/cartpole_mock.yaml --provider mock --timesteps 2000"
