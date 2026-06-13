import csv
import json
from pathlib import Path


def _error_type_from_reason(reason: str) -> str:
    if reason.startswith("pipeline_error"):
        return "pipeline_error"
    if reason.startswith("reflection_error"):
        return "reflection_error"
    if reason.startswith("fallback_score"):
        return "fallback"
    return "none"


def export_memory_csv(memory_jsonl: Path, output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if memory_jsonl.exists():
        with memory_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "generation": row.get("generation"),
                        "candidate_id": row.get("candidate_id"),
                        "score": row.get("judge_score"),
                        "train_return": row.get("train_mean_return"),
                        "error_type": _error_type_from_reason(str(row.get("judge_reason", ""))),
                        "judge_reason": row.get("judge_reason", ""),
                        "video_path": row.get("video_path", ""),
                    }
                )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generation",
                "candidate_id",
                "score",
                "train_return",
                "error_type",
                "judge_reason",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
