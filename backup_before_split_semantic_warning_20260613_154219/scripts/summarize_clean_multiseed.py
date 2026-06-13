#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def infer_env_name(workspace: str) -> str:
    if "cartpole" in workspace.lower():
        return "CartPole-v1"
    if "lunar" in workspace.lower():
        return "LunarLander-v3"
    return "unknown"


def infer_seed(workspace: str) -> str:
    m = re.search(r"seed(\d+)", workspace)
    return m.group(1) if m else ""


def audit_ok(ws: Path) -> bool:
    p = ws / "leak_audit_pre_generation.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("ok", False))
    except Exception:
        return False


def summarize_one(ws: Path) -> dict:
    rows = read_jsonl(ws / "memory.jsonl")
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    statuses = {}
    for r in rows:
        status = str(r.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

    best = None
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r.get("selection_score", -1e18)))

    return {
        "env": infer_env_name(str(ws)),
        "seed": infer_seed(str(ws)),
        "workspace": str(ws),
        "audit_ok": audit_ok(ws),
        "num_records": len(rows),
        "num_ok": len(ok_rows),
        "status_ok": statuses.get("ok", 0),
        "status_invalid_schema": statuses.get("invalid_schema", 0),
        "status_pipeline_error": statuses.get("pipeline_error", 0),
        "repair_attempts_total": sum(int(r.get("repair_attempts", 0) or 0) for r in rows),
        "repair_success_count": sum(int(bool(r.get("repair_success", False))) for r in rows),
        "semantic_warning_count_total": sum(int(r.get("semantic_warning_count", 0) or 0) for r in rows),
        "best_candidate": "" if best is None else best.get("candidate_id"),
        "best_selection_score": "" if best is None else float(best.get("selection_score", 0.0)),
        "best_private_eval_return": "" if best is None else float(best.get("hidden_eval_return", 0.0)),
        "best_generated_return": "" if best is None else float(best.get("train_mean_return", 0.0)),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    envs = sorted(set(r["env"] for r in rows))

    for env in envs:
        group = [r for r in rows if r["env"] == env]
        valid_scores = [
            float(r["best_selection_score"])
            for r in group
            if r["best_selection_score"] != ""
        ]

        out.append(
            {
                "env": env,
                "num_runs": len(group),
                "num_audit_ok": sum(int(bool(r["audit_ok"])) for r in group),
                "num_runs_with_ok_candidate": sum(int(int(r["num_ok"]) > 0) for r in group),
                "mean_best_selection_score": "" if not valid_scores else mean(valid_scores),
                "std_best_selection_score": "" if len(valid_scores) <= 1 else pstdev(valid_scores),
                "mean_num_ok": mean([int(r["num_ok"]) for r in group]) if group else "",
                "mean_invalid_schema": mean([int(r["status_invalid_schema"]) for r in group]) if group else "",
                "mean_pipeline_error": mean([int(r["status_pipeline_error"]) for r in group]) if group else "",
                "mean_repair_attempts": mean([int(r["repair_attempts_total"]) for r in group]) if group else "",
                "mean_repair_success": mean([int(r["repair_success_count"]) for r in group]) if group else "",
                "mean_semantic_warning_count": mean([int(r.get("semantic_warning_count_total", 0)) for r in group]) if group else "",
            }
        )

    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspaces", nargs="+")
    parser.add_argument("--out", type=str, default="runs/clean_multiseed_summary.csv")
    args = parser.parse_args()

    rows = [summarize_one(Path(w)) for w in args.workspaces]
    agg = aggregate(rows)

    out = Path(args.out)
    write_csv(out, rows)

    agg_out = out.with_name(out.stem + "_aggregate.csv")
    write_csv(agg_out, agg)

    print("Per-run summary:")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print()
    print("Aggregate summary:")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print()
    print(f"Saved per-run CSV: {out}")
    print(f"Saved aggregate CSV: {agg_out}")


if __name__ == "__main__":
    main()
