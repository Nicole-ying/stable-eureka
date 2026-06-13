#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_workspace(ws: Path) -> dict:
    rows = read_jsonl(ws / "memory.jsonl")
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    statuses = {}
    for r in rows:
        status = str(r.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

    best = None
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r.get("selection_score", -1e18)))

    private_scores = [float(r.get("hidden_eval_return", 0.0)) for r in ok_rows]
    generated_scores = [float(r.get("train_mean_return", 0.0)) for r in ok_rows]

    audit_ok = False
    audit_path = ws / "leak_audit_pre_generation.json"
    if audit_path.exists():
        try:
            audit_ok = bool(json.loads(audit_path.read_text(encoding="utf-8")).get("ok", False))
        except Exception:
            audit_ok = False

    return {
        "workspace": str(ws),
        "audit_ok": audit_ok,
        "num_records": len(rows),
        "num_ok": len(ok_rows),
        "statuses": json.dumps(statuses, ensure_ascii=False),
        "repair_attempts_total": sum(int(r.get("repair_attempts", 0) or 0) for r in rows),
        "repair_success_count": sum(int(bool(r.get("repair_success", False))) for r in rows),
        "best_candidate": "" if best is None else str(best.get("candidate_id")),
        "best_selection_score": "" if best is None else float(best.get("selection_score", 0.0)),
        "best_private_eval_return": "" if best is None else float(best.get("hidden_eval_return", 0.0)),
        "best_generated_return": "" if best is None else float(best.get("train_mean_return", 0.0)),
        "mean_private_eval_return_ok": "" if not private_scores else mean(private_scores),
        "std_private_eval_return_ok": "" if len(private_scores) <= 1 else pstdev(private_scores),
        "mean_generated_return_ok": "" if not generated_scores else mean(generated_scores),
        "std_generated_return_ok": "" if len(generated_scores) <= 1 else pstdev(generated_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", nargs="+")
    parser.add_argument("--out", type=str, default="runs/clean_summary.csv")
    args = parser.parse_args()

    summaries = [summarize_workspace(Path(w)) for w in args.workspaces]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "workspace",
        "audit_ok",
        "num_records",
        "num_ok",
        "statuses",
        "repair_attempts_total",
        "repair_success_count",
        "best_candidate",
        "best_selection_score",
        "best_private_eval_return",
        "best_generated_return",
        "mean_private_eval_return_ok",
        "std_private_eval_return_ok",
        "mean_generated_return_ok",
        "std_generated_return_ok",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved CSV: {out}")


if __name__ == "__main__":
    main()
