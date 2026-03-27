import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class CandidateRecord:
    generation: int
    candidate_id: str
    parent_ids: list[str]
    reflection_summary: str
    reward_code: str
    llm_rationale: str
    train_mean_return: float
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

    def top_candidates(self, k: int) -> list[dict[str, Any]]:
        rows = self.load_all()
        rows.sort(key=lambda r: r["judge_score"], reverse=True)
        return rows[:k]
