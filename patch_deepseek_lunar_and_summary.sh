#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

mkdir -p mvp/configs scripts

backup_dir="backup_before_deepseek_lunar_summary_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/configs/lunar_lander_clean_deepseek_small.yaml \
  scripts/run_clean_lunar_deepseek_small.sh \
  scripts/summarize_clean_runs.py
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/4] write LunarLander clean DeepSeek config..."
cat > mvp/configs/lunar_lander_clean_deepseek_small.yaml <<'YAML'
model:
  provider: deepseek
  llm_model: deepseek-v4-flash
  vlm_model: deepseek-v4-flash
  deepseek_base_url: https://api.deepseek.com
  deepseek_api_key_env: DEEPSEEK_API_KEY
  temperature: 0.7
  max_tokens: 1200

rl:
  env_id: LunarLander-v3
  total_timesteps: 30000
  eval_episodes: 3
  learning_rate: 0.0003
  gamma: 0.99

evolution:
  generations: 2
  population_size: 2
  elite_size: 1
  reflection_top_k: 2

workspace: runs/clean_lunar_lander_deepseek_g2p2_t30k
seed: 42
YAML

echo "[3/4] write LunarLander clean DeepSeek run script..."
cat > scripts/run_clean_lunar_deepseek_small.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="runs/clean_lunar_lander_deepseek_g2p2_t30k"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set."
  echo "Please run:"
  echo "  export DEEPSEEK_API_KEY='your_key_here'"
  exit 1
fi

rm -rf "$WORKSPACE"

python run_mvp.py \
  --config mvp/configs/lunar_lander_clean_deepseek_small.yaml

python scripts/audit_clean_run.py "$WORKSPACE"
SH

chmod +x scripts/run_clean_lunar_deepseek_small.sh

echo "[4/4] write clean run summarizer..."
cat > scripts/summarize_clean_runs.py <<'PY'
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
PY

chmod +x scripts/summarize_clean_runs.py
python -m py_compile scripts/summarize_clean_runs.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Next:"
echo "  bash scripts/run_clean_lunar_deepseek_small.sh"
echo ""
echo "Then summarize:"
echo "  python scripts/summarize_clean_runs.py runs/clean_cartpole_deepseek_g2p2_t8k runs/clean_lunar_lander_deepseek_g2p2_t30k"
