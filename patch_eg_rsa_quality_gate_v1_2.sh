#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] check repo root..."
test -d mvp || { echo "ERROR: run this script at repo root"; exit 1; }

backup_dir="backup_before_quality_gate_v1_2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

if [ -f mvp/prompts/lesson_extractor_system.txt ]; then
  mkdir -p "$backup_dir/mvp/prompts"
  cp mvp/prompts/lesson_extractor_system.txt "$backup_dir/mvp/prompts/lesson_extractor_system.txt"
fi

echo "[2/3] patch lesson extractor prompt..."
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
- np and math are available in the reward execution namespace.
- Import statements are forbidden, but using np or math without importing them is allowed.
- Only flag next_obs usage if it causes a concrete issue such as unstable scaling, incorrect indexing, or private-signal leakage.
- Only flag np/math usage if the generated code tries to import them or uses unsupported functions.

Validation interpretation:
- The validator may use a generic smoke-test observation vector.
- If unpacking fails with "too many values to unpack", recommend slicing obs[:N] or next_obs[:N] before unpacking.
- Do not claim the real environment necessarily has extra observation dimensions unless the task files show that.

For candidate scope:
- Focus on candidate-specific causes of success/failure.
- Mention component imbalance, inactive components, generated/private gap, action collapse, repair/validation problems, or useful mutations.
- Do not invent availability constraints that contradict the reward contract.

For environment scope:
- Focus on recurring patterns across candidates in this generation.

Do not include code blocks.
TXT

echo "[3/3] syntax checks..."
python -m py_compile mvp/*.py scripts/check_run_quality.py scripts/check_eureka_step_input.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
