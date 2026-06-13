import csv
import json
from pathlib import Path


def _error_type_from_reason(reason: str) -> str:
    if reason.startswith("pipeline_error"):
        return "pipeline_error"
    if reason.startswith("validation_error"):
        return "validation_error"
    if reason.startswith("reflection_error"):
        return "reflection_error"
    if reason.startswith("visual_judge_error"):
        return "visual_judge_error"
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
                        "schema_version": row.get("schema_version"),
                        "env_alias": row.get("env_alias"),
                        "status": row.get("status"),
                        "selection_score": row.get("selection_score"),
                        "private_eval_return": row.get("hidden_eval_return"),
                        "generated_return": row.get("train_mean_return"),
                        "repair_attempts": row.get("repair_attempts", 0),
                        "repair_success": row.get("repair_success", False),
                        "judge_score": row.get("judge_score"),
                        "error_type": _error_type_from_reason(str(row.get("judge_reason", ""))),
                        "validation_errors": row.get("validation_errors", []),
                        "validation_errors_before_repair": row.get("validation_errors_before_repair", []),
                        "validation_errors_after_repair": row.get("validation_errors_after_repair", []),
                        "judge_reason": row.get("judge_reason", ""),
                        "video_path": row.get("video_path", ""),
                    }
                )

    fieldnames = [
        "generation",
        "candidate_id",
        "schema_version",
        "env_alias",
        "status",
        "selection_score",
        "private_eval_return",
        "generated_return",
        "repair_attempts",
        "repair_success",
        "judge_score",
        "error_type",
        "validation_errors",
        "validation_errors_before_repair",
        "validation_errors_after_repair",
        "judge_reason",
        "video_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv
