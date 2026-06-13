#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_split_semantic_warning_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/semantic_audit.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/orchestrator.py \
  scripts/audit_clean_run.py \
  scripts/summarize_clean_multiseed.py
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/7] rewrite mvp/semantic_audit.py..."
cat > mvp/semantic_audit.py <<'PY'
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# Semantic warning audit
# ============================================================
#
# 设计原则：
#   1. 这里只做 warning，不做 hard fail；
#   2. 不直接影响训练、选择、validator；
#   3. 用于统计 LLM 是否在匿名接口下仍然推断 benchmark 身份
#      或给 obs 维度赋予物理语义；
#   4. 不写某个环境的专用逻辑，词表覆盖常见控制任务。
#
# 两类 warning：
#   A. identity warnings:
#      疑似 benchmark / agent / object identity 推断。
#
#   B. physical semantic warnings:
#      疑似 observation dimension physical semantics 推断。
#
# 注意：
#   - "goal" 太通用，已移除，避免 clean_plan 里 Task goal 字段误报；
#   - "progress/stability/effort/terminal" 是 schema 词，不在这里统计。
# ============================================================


IDENTITY_WARNING_TERMS = (
    # classic control / gym-like benchmark identity
    "cart",
    "pole",
    "cartpole",
    "lander",
    "lunar",
    "landing",
    "mountain",
    "mountaincar",
    "pendulum",
    "acrobot",

    # locomotion / robot-like identity
    "walker",
    "bipedal",
    "hopper",
    "cheetah",
    "ant",
    "humanoid",
    "robot",

    # task object identity
    "leg",
    "legs",
    "contact",
    "contacts",
    "thruster",
    "engine",
    "engines",
)


PHYSICAL_SEMANTIC_WARNING_TERMS = (
    # coordinate / state semantics
    "position",
    "positions",
    "velocity",
    "velocities",
    "angle",
    "angles",
    "angular",
    "coordinate",
    "coordinates",
    "x coordinate",
    "y coordinate",
    "height",
    "altitude",
    "distance",
    "origin",
    "target",

    # dynamics / behavior semantics
    "upright",
    "balance",
    "balancing",
    "fall",
    "falling",
    "crash",
    "land",
    "landed",
    "success flag",
    "failure",
    "torque",
    "speed",
    "acceleration",
)


def _count_terms(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    lower = text.lower()
    counts: dict[str, int] = {}

    for term in terms:
        term_lower = term.lower()

        if " " in term_lower:
            n = lower.count(term_lower)
        else:
            # Use word boundary to reduce false positives.
            n = len(re.findall(rf"\b{re.escape(term_lower)}\b", lower))

        if n:
            counts[term] = n

    return counts


def _audit_term_group(bundle: dict[str, Any], terms: tuple[str, ...]) -> tuple[int, dict[str, dict[str, int]]]:
    per_artifact: dict[str, dict[str, int]] = {}
    total = 0

    for name, value in bundle.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        counts = _count_terms(text, terms)
        per_artifact[name] = counts
        total += sum(counts.values())

    return total, per_artifact


def audit_semantic_text_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Return semantic warning statistics.

    Backward-compatible fields:
      - semantic_warning_count
      - semantic_warnings

    New split fields:
      - identity_warning_count / identity_warnings
      - semantic_term_warning_count / semantic_term_warnings
    """
    identity_total, identity_warnings = _audit_term_group(bundle, IDENTITY_WARNING_TERMS)
    semantic_total, semantic_warnings = _audit_term_group(bundle, PHYSICAL_SEMANTIC_WARNING_TERMS)

    merged: dict[str, dict[str, int]] = {}
    for name in set(identity_warnings) | set(semantic_warnings):
        merged[name] = {}
        merged[name].update(identity_warnings.get(name, {}))
        merged[name].update(semantic_warnings.get(name, {}))

    return {
        "identity_warning_count": identity_total,
        "identity_warnings": identity_warnings,
        "semantic_term_warning_count": semantic_total,
        "semantic_term_warnings": semantic_warnings,

        # Backward-compatible total fields.
        "semantic_warning_count": identity_total + semantic_total,
        "semantic_warnings": merged,
    }


def save_semantic_audit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[3/7] patch mvp/memory.py..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/memory.py")
s = p.read_text(encoding="utf-8")

old = """    semantic_warning_count: int
    semantic_warnings: dict[str, Any]

    reflection_summary: str
"""

new = """    identity_warning_count: int
    identity_warnings: dict[str, Any]
    semantic_term_warning_count: int
    semantic_term_warnings: dict[str, Any]
    semantic_warning_count: int
    semantic_warnings: dict[str, Any]

    reflection_summary: str
"""

if old not in s:
    if "identity_warning_count" in s:
        print("memory.py already has split warning fields; skip.")
    else:
        raise SystemExit("ERROR: expected semantic_warning_count block not found in mvp/memory.py")
else:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
PY

echo "[4/7] rewrite mvp/exporters.py..."
cat > mvp/exporters.py <<'PY'
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

                        "identity_warning_count": row.get("identity_warning_count", 0),
                        "identity_warnings": row.get("identity_warnings", {}),
                        "semantic_term_warning_count": row.get("semantic_term_warning_count", 0),
                        "semantic_term_warnings": row.get("semantic_term_warnings", {}),

                        # Backward-compatible total semantic warning fields.
                        "semantic_warning_count": row.get("semantic_warning_count", 0),
                        "semantic_warnings": row.get("semantic_warnings", {}),

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

        "identity_warning_count",
        "identity_warnings",
        "semantic_term_warning_count",
        "semantic_term_warnings",
        "semantic_warning_count",
        "semantic_warnings",

        "judge_reason",
        "video_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv
PY

echo "[5/7] patch mvp/orchestrator.py..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/orchestrator.py")
s = p.read_text(encoding="utf-8")

old_record = """                    semantic_warning_count=int(semantic_report.get("semantic_warning_count", 0)),
                    semantic_warnings=semantic_report.get("semantic_warnings", {}),
                    reflection_summary=reflection,
"""

new_record = """                    identity_warning_count=int(semantic_report.get("identity_warning_count", 0)),
                    identity_warnings=semantic_report.get("identity_warnings", {}),
                    semantic_term_warning_count=int(semantic_report.get("semantic_term_warning_count", 0)),
                    semantic_term_warnings=semantic_report.get("semantic_term_warnings", {}),
                    semantic_warning_count=int(semantic_report.get("semantic_warning_count", 0)),
                    semantic_warnings=semantic_report.get("semantic_warnings", {}),
                    reflection_summary=reflection,
"""

if old_record not in s:
    if "identity_warning_count=int(semantic_report.get" in s:
        print("orchestrator.py already has split warning fields in CandidateRecord; skip record patch.")
    else:
        raise SystemExit("ERROR: CandidateRecord semantic block not found in orchestrator.py")
else:
    s = s.replace(old_record, new_record)

old_report = """        f"repair_success: {best.get('repair_success', False)}",
        f"semantic_warning_count: {best.get('semantic_warning_count', 0)}",
        f"judge_score: {best.get('judge_score', 0)}",
"""

new_report = """        f"repair_success: {best.get('repair_success', False)}",
        f"identity_warning_count: {best.get('identity_warning_count', 0)}",
        f"semantic_term_warning_count: {best.get('semantic_term_warning_count', 0)}",
        f"semantic_warning_count: {best.get('semantic_warning_count', 0)}",
        f"judge_score: {best.get('judge_score', 0)}",
"""

if old_report in s:
    s = s.replace(old_report, new_report)
elif "identity_warning_count:" in s:
    print("orchestrator.py report already has split warning fields; skip report patch.")
else:
    raise SystemExit("ERROR: report semantic block not found in orchestrator.py")

p.write_text(s, encoding="utf-8")
PY

echo "[6/7] rewrite scripts/audit_clean_run.py and patch summarizer..."
cat > scripts/audit_clean_run.py <<'PY'
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
PY

chmod +x scripts/audit_clean_run.py

python - <<'PY'
from pathlib import Path

p = Path("scripts/summarize_clean_multiseed.py")
if not p.exists():
    print("scripts/summarize_clean_multiseed.py not found; skip.")
    raise SystemExit(0)

s = p.read_text(encoding="utf-8")

# Add per-run fields after repair_success_count if not present.
if '"identity_warning_count_total":' not in s:
    s = s.replace(
        '        "repair_success_count": sum(int(bool(r.get("repair_success", False))) for r in rows),\n',
        '        "repair_success_count": sum(int(bool(r.get("repair_success", False))) for r in rows),\n'
        '        "identity_warning_count_total": sum(int(r.get("identity_warning_count", 0) or 0) for r in rows),\n'
        '        "semantic_term_warning_count_total": sum(int(r.get("semantic_term_warning_count", 0) or 0) for r in rows),\n'
        '        "semantic_warning_count_total": sum(int(r.get("semantic_warning_count", 0) or 0) for r in rows),\n',
    )

# Add aggregate fields if not present.
if '"mean_identity_warning_count":' not in s:
    s = s.replace(
        '                "mean_repair_success": mean([int(r["repair_success_count"]) for r in group]) if group else "",\n',
        '                "mean_repair_success": mean([int(r["repair_success_count"]) for r in group]) if group else "",\n'
        '                "mean_identity_warning_count": mean([int(r.get("identity_warning_count_total", 0)) for r in group]) if group else "",\n'
        '                "mean_semantic_term_warning_count": mean([int(r.get("semantic_term_warning_count_total", 0)) for r in group]) if group else "",\n'
        '                "mean_semantic_warning_count": mean([int(r.get("semantic_warning_count_total", 0)) for r in group]) if group else "",\n',
    )

p.write_text(s, encoding="utf-8")
PY

echo "[7/7] syntax check..."
python -m py_compile \
  mvp/semantic_audit.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/orchestrator.py \
  scripts/audit_clean_run.py

if [ -f scripts/summarize_clean_multiseed.py ]; then
  python -m py_compile scripts/summarize_clean_multiseed.py
fi

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Next small verification:"
echo "  rm -rf runs/clean_cartpole_deepseek_seed0_g2p2_t8k"
echo "  python run_mvp.py --config mvp/configs/cartpole_clean_deepseek_seed0.yaml"
echo "  python scripts/audit_clean_run.py runs/clean_cartpole_deepseek_seed0_g2p2_t8k"
echo ""
echo "Then rerun full anonymized multiseed:"
echo "  bash scripts/run_clean_cartpole_deepseek_multiseed.sh"
echo "  bash scripts/run_clean_lunar_deepseek_multiseed.sh"
