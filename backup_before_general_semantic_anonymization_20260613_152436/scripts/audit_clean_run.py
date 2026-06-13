#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


LEAK_TERMS = (
    "env_reward",
    "hidden_env_reward",
    "_hidden_env_reward",
    "fitness_score",
    "compute_fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
    "LunarLander",
    "BipedalWalker",
    "CartPole",
    "Acrobot",
    "MountainCar",
    "Pendulum",
)


PROMPT_ARTIFACTS = (
    "clean_interface.txt",
    "reward_schema.txt",
    "clean_plan.txt",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def check_text_artifact(path: Path) -> list[dict]:
    if not path.exists():
        return [{"artifact": str(path), "term": "<missing_file>"}]

    text = path.read_text(encoding="utf-8")
    lower = text.lower()

    out = []
    for term in LEAK_TERMS:
        if term.lower() in lower:
            out.append({"artifact": str(path), "term": term})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=str)
    args = parser.parse_args()

    ws = Path(args.workspace)
    if not ws.exists():
        raise SystemExit(f"workspace does not exist: {ws}")

    report = {
        "workspace": str(ws),
        "pre_generation_audit_ok": None,
        "prompt_artifact_violations": [],
        "memory_summary": {},
        "recommendation": "",
    }

    audit_path = ws / "leak_audit_pre_generation.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        report["pre_generation_audit_ok"] = bool(audit.get("ok", False))
        if not audit.get("ok", False):
            report["prompt_artifact_violations"].extend(audit.get("violations", []))
    else:
        report["pre_generation_audit_ok"] = False
        report["prompt_artifact_violations"].append(
            {"artifact": str(audit_path), "term": "<missing_audit_report>"}
        )

    for name in PROMPT_ARTIFACTS:
        report["prompt_artifact_violations"].extend(check_text_artifact(ws / name))

    rows = read_jsonl(ws / "memory.jsonl")
    statuses = {}
    repair_attempts = 0
    repair_success = 0

    for row in rows:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
        repair_attempts += int(row.get("repair_attempts", 0) or 0)
        repair_success += int(bool(row.get("repair_success", False)))

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    best = None
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r.get("selection_score", -1e18)))

    report["memory_summary"] = {
        "num_records": len(rows),
        "statuses": statuses,
        "num_ok": len(ok_rows),
        "repair_attempts_total": repair_attempts,
        "repair_success_count": repair_success,
        "best_candidate": None
        if best is None
        else {
            "candidate_id": best.get("candidate_id"),
            "selection_score": best.get("selection_score"),
            "private_eval_return": best.get("hidden_eval_return"),
            "generated_return": best.get("train_mean_return"),
            "repair_attempts": best.get("repair_attempts"),
            "repair_success": best.get("repair_success"),
        },
    }

    if report["prompt_artifact_violations"]:
        report["recommendation"] = "FAIL: prompt-facing artifacts contain leak terms or missing files."
    elif len(rows) == 0:
        report["recommendation"] = "FAIL: no memory records were produced; pipeline may not have run."
    elif not ok_rows:
        report["recommendation"] = "CHECK: no valid trained candidates; inspect validation/repair failures."
    else:
        report["recommendation"] = "PASS: clean run artifacts look usable for the next-stage experiment."

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
