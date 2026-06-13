#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter


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
    if not path.exists():
        return []

    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def find_prompt_artifact_violations(workspace: Path) -> list[dict]:
    violations = []

    for rel in PROMPT_ARTIFACTS:
        p = workspace / rel
        if not p.exists():
            violations.append({"artifact": rel, "error": "missing"})
            continue

        text = p.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()

        hits = []
        for term in LEAK_TERMS:
            if term.lower() in lower:
                hits.append(term)

        if hits:
            violations.append({"artifact": rel, "hits": sorted(set(hits))})

    return violations


def summarize_memory(workspace: Path) -> dict:
    rows = read_jsonl(workspace / "memory.jsonl")
    statuses = Counter(str(r.get("status", "unknown")) for r in rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    best = None
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r.get("selection_score", -1e18)))

    return {
        "num_records": len(rows),
        "statuses": dict(statuses),
        "num_ok": len(ok_rows),
        "repair_attempts_total": sum(int(r.get("repair_attempts", 0) or 0) for r in rows),
        "repair_success_count": sum(int(bool(r.get("repair_success", False))) for r in rows),

        "identity_warning_count_total": sum(int(r.get("identity_warning_count", 0) or 0) for r in rows),
        "semantic_term_warning_count_total": sum(int(r.get("semantic_term_warning_count", 0) or 0) for r in rows),
        "semantic_warning_count_total": sum(int(r.get("semantic_warning_count", 0) or 0) for r in rows),

        "best_candidate": None if best is None else {
            "candidate_id": best.get("candidate_id"),
            "selection_score": best.get("selection_score"),
            "private_eval_return": best.get("hidden_eval_return"),
            "generated_return": best.get("train_mean_return"),
            "identity_warning_count": best.get("identity_warning_count", 0),
            "semantic_term_warning_count": best.get("semantic_term_warning_count", 0),
            "semantic_warning_count": best.get("semantic_warning_count", 0),
        },
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/audit_clean_run.py <workspace>")
        raise SystemExit(2)

    workspace = Path(sys.argv[1])

    pre_generation_audit_ok = False
    audit_path = workspace / "leak_audit_pre_generation.json"
    if audit_path.exists():
        try:
            pre_generation_audit_ok = bool(json.loads(audit_path.read_text(encoding="utf-8")).get("ok", False))
        except Exception:
            pre_generation_audit_ok = False

    semantic_pre_generation = None
    semantic_path = workspace / "semantic_audit_pre_generation.json"
    if semantic_path.exists():
        try:
            semantic_pre_generation = json.loads(semantic_path.read_text(encoding="utf-8"))
        except Exception as e:
            semantic_pre_generation = {"error": str(e)}

    prompt_artifact_violations = find_prompt_artifact_violations(workspace)
    memory_summary = summarize_memory(workspace)

    if prompt_artifact_violations:
        recommendation = "FAIL: prompt-facing artifacts contain leak terms or missing files."
    elif memory_summary["num_records"] == 0:
        recommendation = "FAIL: no memory records; run did not generate candidates."
    elif memory_summary["num_ok"] == 0:
        recommendation = "CHECK: no valid trained candidates; inspect validation/repair failures."
    else:
        recommendation = "PASS: clean run artifacts look usable. Inspect semantic warning counts for observation semantic inference."

    report = {
        "workspace": str(workspace),
        "pre_generation_audit_ok": pre_generation_audit_ok,
        "prompt_artifact_violations": prompt_artifact_violations,
        "semantic_pre_generation": semantic_pre_generation,
        "memory_summary": memory_summary,
        "recommendation": recommendation,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
