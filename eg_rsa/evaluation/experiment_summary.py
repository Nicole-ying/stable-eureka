from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentSummary:
    """Aggregate EG-RSA iteration outputs into JSON and CSV summaries."""

    @staticmethod
    def build(output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        rows: List[Dict[str, Any]] = []
        for iter_dir in sorted(output_dir.glob("iteration_*")):
            if not iter_dir.is_dir():
                continue
            row = ExperimentSummary._read_iteration(iter_dir)
            rows.append(row)

        summary = {
            "output_dir": str(output_dir),
            "num_iterations": len(rows),
            "iterations": rows,
            "best_by_task_score": ExperimentSummary._best_row(rows, "task_score"),
            "best_by_posthoc_return": ExperimentSummary._best_row(rows, "posthoc_return_mean"),
        }
        return summary

    @staticmethod
    def save(output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        summary = ExperimentSummary.build(output_dir)
        ExperimentSummary._write_json(output_dir / "summary.json", summary)
        ExperimentSummary._write_csv(output_dir / "summary.csv", summary.get("iterations", []))
        return summary

    @staticmethod
    def _read_iteration(iter_dir: Path) -> Dict[str, Any]:
        idx = ExperimentSummary._iteration_index(iter_dir)
        diagnostic = ExperimentSummary._read_json(iter_dir / "diagnostic_report.json", {})
        posthoc = ExperimentSummary._read_json(iter_dir / "posthoc_eval.json", {})
        edit_validation = ExperimentSummary._read_json(iter_dir / "edit_validation.json", {})
        edit_plan = ExperimentSummary._read_json(iter_dir / "edit_plan.json", {})
        memory_card = ExperimentSummary._read_json(iter_dir / "memory_card.json", {})

        diagnostics = diagnostic.get("diagnostics", {})
        attribution = diagnostic.get("attribution", {})
        row = {
            "iteration": idx,
            "task_score": None,
            "hack_score": diagnostics.get("hack_score"),
            "failure_modes": ";".join(diagnostics.get("failure_modes", [])),
            "dominant_component": diagnostics.get("dominant_component"),
            "dominant_component_ratio": diagnostics.get("dominant_component_ratio"),
            "posthoc_return_mean": posthoc.get("return_mean"),
            "posthoc_return_std": posthoc.get("return_std"),
            "posthoc_episode_length_mean": posthoc.get("episode_length_mean"),
            "edit_valid": edit_validation.get("is_valid"),
            "edit_errors": ";".join(edit_validation.get("errors", [])),
            "edit_count": len(edit_plan.get("edit_plan", [])),
            "memory_failure_modes": ";".join(memory_card.get("failure_modes", [])),
        }
        row.update(ExperimentSummary._task_score_from_memory(memory_card))
        row.update(ExperimentSummary._component_ratios(attribution))
        return row

    @staticmethod
    def _task_score_from_memory(memory_card: Dict[str, Any]) -> Dict[str, Any]:
        outcome = memory_card.get("outcome", {})
        return {"task_score": outcome.get("task_score_before")}

    @staticmethod
    def _component_ratios(attribution: Dict[str, Any]) -> Dict[str, Any]:
        stats = attribution.get("component_stats", {})
        out: Dict[str, Any] = {}
        for name, values in stats.items():
            safe = name.replace(" ", "_")
            out[f"component_ratio__{safe}"] = values.get("ratio")
            out[f"component_trigger__{safe}"] = values.get("trigger_rate")
        return out

    @staticmethod
    def _best_row(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        valid = [row for row in rows if row.get(key) is not None]
        if not valid:
            return {}
        return max(valid, key=lambda row: float(row.get(key)))

    @staticmethod
    def _iteration_index(iter_dir: Path) -> int:
        try:
            return int(iter_dir.name.split("_")[-1])
        except ValueError:
            return -1

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
