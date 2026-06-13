#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] check repo root..."
test -d mvp || { echo "ERROR: run this script at repo root"; exit 1; }

backup_dir="backup_before_quality_gate_v1_3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/lessons.py \
  mvp/prompts/lesson_extractor_system.txt \
  scripts/check_run_quality.py
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/5] patch lesson extractor prompt..."
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

Private evaluator boundary:
- private_eval_return is a black-box selection and diagnostic signal only.
- You may say a reward candidate has high or low private_eval_return.
- You may say generated_return and private_eval_return are misaligned.
- Do not recommend reconstructing, inferring, investigating, reverse engineering, approximating, or imitating the hidden evaluator formula or structure.
- Do not write recommendations such as "investigate the hidden evaluator's structure".
- Prefer wording such as "use black-box selection feedback to reduce generated/private mismatch."

Action-space boundary:
- Do not guess whether the action space is continuous or discrete.
- If the task files show discrete action IDs, use those IDs.
- If the task files show continuous actions, describe continuous magnitudes.
- Do not write "continuous action is common" unless the provided step.py actually uses continuous actions for this run.

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

echo "[3/5] patch lessons.py with lesson sanitizer..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/lessons.py")
s = p.read_text(encoding="utf-8")

insert = r'''
UNSAFE_LESSON_PHRASES = (
    "hidden evaluator's likely structure",
    "hidden evaluator structure",
    "hidden evaluator formula",
    "infer the hidden evaluator",
    "infer hidden evaluator",
    "reconstruct the hidden evaluator",
    "reverse engineer",
    "imitate the hidden evaluator",
    "approximate the hidden evaluator",
)

ACTION_SPACE_GUESS_PHRASES = (
    "continuous action is common",
    "common in lunarlender",
    "common in lunarlander",
    "if actions are continuous",
)


def sanitize_lesson_text(row: dict[str, Any]) -> dict[str, Any]:
    """
    Remove unsafe or misleading lesson wording before storing memory.

    Lessons may use private_eval_return as black-box feedback, but must not
    recommend reconstructing hidden evaluator internals.
    """
    out = dict(row)

    text_fields = ["condition", "observation", "explanation", "recommendation"]
    combined = " ".join(str(out.get(k, "")) for k in text_fields).lower()

    if any(p.lower() in combined for p in UNSAFE_LESSON_PHRASES):
        out["recommendation"] = (
            "Use private_eval_return only as black-box selection feedback. "
            "Reduce generated/private mismatch by adjusting reward component scales and behavior diagnostics, "
            "without inferring or reconstructing hidden evaluator internals."
        )
        out["confidence"] = min(float(out.get("confidence", 0.5)), 0.5)
        out["lesson_type"] = "prompt_rule"

    combined = " ".join(str(out.get(k, "")) for k in text_fields).lower()
    if any(p.lower() in combined for p in ACTION_SPACE_GUESS_PHRASES):
        out["recommendation"] = (
            "Use only the action semantics explicitly shown in the provided step.py. "
            "Do not guess continuous or discrete action structure beyond the task files."
        )
        out["confidence"] = min(float(out.get("confidence", 0.5)), 0.6)
        out["lesson_type"] = "prompt_rule"

    return out
'''

if "def sanitize_lesson_text" not in s:
    s = s.replace("def read_jsonl", insert + "\n\ndef read_jsonl")

old = '''    out = dict(lesson)
    out.setdefault("lesson_id", f"{scope}_{uuid.uuid4().hex[:10]}")
'''

new = '''    out = sanitize_lesson_text(dict(lesson))
    out.setdefault("lesson_id", f"{scope}_{uuid.uuid4().hex[:10]}")
'''

if old not in s:
    raise SystemExit("Could not patch normalize_lesson. Please inspect mvp/lessons.py manually.")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
PY

echo "[4/5] patch check_run_quality.py with unsafe lesson checks..."
python - <<'PY'
from pathlib import Path

p = Path("scripts/check_run_quality.py")
s = p.read_text(encoding="utf-8")

needle = '''    env_lessons = read_jsonl(run_dir / "env_lessons.jsonl")
    if memory_rows and not env_lessons:
        errors.append("missing env_lessons.jsonl or no environment lessons generated")
'''

replacement = '''    env_lessons = read_jsonl(run_dir / "env_lessons.jsonl")
    if memory_rows and not env_lessons:
        errors.append("missing env_lessons.jsonl or no environment lessons generated")

    unsafe_phrases = [
        "hidden evaluator's likely structure",
        "hidden evaluator structure",
        "hidden evaluator formula",
        "infer the hidden evaluator",
        "reconstruct the hidden evaluator",
        "reverse engineer",
        "imitate the hidden evaluator",
        "approximate the hidden evaluator",
    ]
    action_guess_phrases = [
        "continuous action is common",
        "if actions are continuous",
    ]

    for source_name, rows in [
        ("candidate_lessons", candidate_lessons),
        ("env_lessons", env_lessons),
    ]:
        for row in rows:
            text = " ".join(str(row.get(k, "")) for k in ("condition", "observation", "explanation", "recommendation")).lower()
            if any(p.lower() in text for p in unsafe_phrases):
                errors.append(f"unsafe hidden-evaluator wording in {source_name}: {row.get('lesson_id')}")
            if any(p.lower() in text for p in action_guess_phrases):
                warnings.append(f"possible action-space guess in {source_name}: {row.get('lesson_id')}")
'''

if needle not in s:
    raise SystemExit("Could not patch check_run_quality.py. Please inspect manually.")

s = s.replace(needle, replacement)
p.write_text(s, encoding="utf-8")
PY

echo "[5/5] syntax checks..."
python -m py_compile mvp/*.py scripts/check_run_quality.py scripts/check_eureka_step_input.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Next:"
echo "  python scripts/check_run_quality.py runs/eg_rsa_lunar_deepseek_seed0_g2p2_t30k"
echo ""
echo "Note: old run may still warn/error because lessons were generated before sanitizer."
echo "Future runs will sanitize lessons before writing."
