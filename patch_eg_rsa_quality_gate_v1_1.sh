#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] check repo root..."
test -d mvp || { echo "ERROR: run this script at repo root"; exit 1; }

backup_dir="backup_before_quality_gate_v1_1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/reward_schema.py \
  mvp/prompts/lesson_extractor_system.txt
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/4] patch reward_schema.py: force schema_version to reflect normalized components..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/reward_schema.py")
s = p.read_text(encoding="utf-8")

old = '''    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    schema["schema_version"] = str(schema.get("schema_version") or f"eg_rsa_reward_schema_v1_{schema_hash}")
    if not schema["schema_version"].startswith("eg_rsa_reward_schema"):
        schema["schema_version"] = f"eg_rsa_reward_schema_v1_{schema_hash}"

    return schema
'''

new = '''    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

    # Quality-gate v1.1:
    # Schema version must reflect the final normalized component set.
    # Do not preserve a default schema_version after LLM components are normalized,
    # otherwise different schemas can share the same version and contaminate memory retrieval.
    schema["schema_version"] = f"eg_rsa_reward_schema_v1_{schema_hash}"

    return schema
'''

if old not in s:
    raise SystemExit("Could not find schema_version block. Patch manually.")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
PY

echo "[3/4] patch lesson extractor prompt: next_obs is valid input..."
cat > mvp/prompts/lesson_extractor_system.txt <<'TXT'
You extract reusable lessons from reward-search evidence and reflection reports.

Return JSON array only.
Each item should contain:
{
  "lesson_type": "reward_pattern | failure_mode | mutation_rule | repair_rule | prompt_rule | general",
  "condition": "...",
  "observation": "...",
  "explanation": "...",
  "recommendation": "...",
  "confidence": 0.0,
  "reuse_policy": "same_env | similar_env | global"
}

Reward input boundary:
- compute_reward receives obs, action, next_obs, done, info.
- obs is valid.
- action is valid.
- next_obs is valid.
- done is valid.
- info is valid only if it does not contain private evaluator or hidden reward fields.
- Do not claim that next_obs is unavailable.
- Only flag next_obs usage if it causes a concrete issue such as unstable scaling, incorrect indexing, or private-signal leakage.

For candidate scope:
- Focus on candidate-specific causes of success/failure.
- Mention component imbalance, inactive components, generated/private gap, action collapse, repair/validation problems, or useful mutations.
- Do not invent availability constraints that contradict the reward contract.

For environment scope:
- Focus on recurring patterns across candidates in this generation.

Do not include code blocks.
TXT

echo "[4/4] syntax checks..."
python -m py_compile mvp/*.py scripts/check_run_quality.py scripts/check_eureka_step_input.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Next:"
echo "  rm -rf runs/eg_rsa_lunar_deepseek_smoke"
echo "  bash scripts/run_eg_rsa_lunar_smoke.sh"
echo "  python scripts/check_run_quality.py runs/eg_rsa_lunar_deepseek_smoke"
