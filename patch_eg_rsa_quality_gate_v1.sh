#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] check repo root..."
test -d mvp || { echo "ERROR: run this script at repo root"; exit 1; }

backup_dir="backup_before_quality_gate_v1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/lessons.py \
  mvp/orchestrator.py \
  mvp/prompts/schema_planner_system.txt \
  mvp/prompts/reward_coder_system.txt \
  mvp/prompts/lesson_extractor_system.txt \
  scripts/check_run_quality.py
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/7] patch reward_schema.py: no blind default-component appending..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/reward_schema.py")
s = p.read_text(encoding="utf-8")

old = '''def normalize_schema(raw: dict[str, Any] | None, clean_interface: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    default = build_default_schema(clean_interface)

    schema = dict(default)
    schema.update({k: v for k, v in raw.items() if v is not None})

    components = raw.get("components")
    if not isinstance(components, list) or not components:
        components = default["components"]

    normalized_components = []
    seen = set()
    for c in components:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", "")).strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        normalized_components.append(
            {
                "id": cid,
                "description": str(c.get("description", f"{cid} component")),
                "direction": str(c.get("direction", "maximize")),
                "required": bool(c.get("required", True)),
            }
        )

    required_ids = {c["id"] for c in normalized_components if c.get("required")}
    for c in default["components"]:
        if c["id"] not in seen:
            normalized_components.append(c)
            required_ids.add(c["id"])

    schema["components"] = normalized_components
    schema["reward_signature"] = "compute_reward(obs, action, next_obs, done, info)"
    schema["return_contract"] = "return float(total_reward), components_dict"
    schema["allowed_inputs"] = REQUIRED_SIGNATURE
    schema["reward_abs_bound"] = float(schema.get("reward_abs_bound", 1000.0))

    payload = {
        "env_alias": clean_interface.get("env_alias"),
        "components": schema["components"],
        "task_head": clean_interface.get("eureka_task_description", "")[:1000],
    }
    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    schema["schema_version"] = str(schema.get("schema_version") or f"eg_rsa_reward_schema_v1_{schema_hash}")
    if not schema["schema_version"].startswith("eg_rsa_reward_schema"):
        schema["schema_version"] = f"eg_rsa_reward_schema_v1_{schema_hash}"

    return schema
'''

new = '''def normalize_schema(raw: dict[str, Any] | None, clean_interface: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize an LLM-generated reward schema.

    Quality-gate v1 policy:
      - If the LLM provides at least 4 valid components, trust that schema.
      - Do not blindly append default progress/stability/effort/terminal.
      - Keep required component count between 4 and 6 where possible.
      - Avoid multiple terminal-like components activating on done.
      - Fall back to default schema only when LLM schema is missing or too small.
    """
    if not isinstance(raw, dict):
        raw = {}

    default = build_default_schema(clean_interface)

    schema = dict(default)
    schema.update({k: v for k, v in raw.items() if v is not None})

    raw_components = raw.get("components")
    if not isinstance(raw_components, list):
        raw_components = []

    normalized_components = []
    seen = set()
    terminal_like_seen = False

    for c in raw_components:
        if not isinstance(c, dict):
            continue

        cid = str(c.get("id", "")).strip()
        if not cid:
            continue

        cid_norm = cid.lower()
        is_terminal_like = any(
            term in cid_norm
            for term in ("terminal", "landing", "crash", "success", "failure", "done")
        )

        # Keep one terminal-outcome component only. This prevents schemas such as
        # landing_bonus + crash_penalty + terminal all being required simultaneously.
        if is_terminal_like:
            if terminal_like_seen:
                continue
            cid = "terminal"
            terminal_like_seen = True

        if cid in seen:
            continue

        seen.add(cid)
        normalized_components.append(
            {
                "id": cid,
                "description": str(c.get("description", f"{cid} component")),
                "direction": str(c.get("direction", "maximize")),
                "required": bool(c.get("required", True)),
            }
        )

    # If LLM schema is too small or malformed, use default compact schema.
    if len(normalized_components) < 4:
        normalized_components = list(default["components"])
    else:
        # Cap at 6 required components to keep reward code compact and interpretable.
        required = [c for c in normalized_components if c.get("required", True)]
        optional = [c for c in normalized_components if not c.get("required", True)]

        if len(required) > 6:
            kept_required = required[:6]
            kept_ids = {c["id"] for c in kept_required}
            normalized_components = kept_required + [c for c in optional if c["id"] in kept_ids]
        else:
            normalized_components = normalized_components[:6]

    schema["components"] = normalized_components
    schema["reward_signature"] = "compute_reward(obs, action, next_obs, done, info)"
    schema["return_contract"] = "return float(total_reward), components_dict"
    schema["allowed_inputs"] = REQUIRED_SIGNATURE
    schema["reward_abs_bound"] = float(schema.get("reward_abs_bound", 1000.0))

    payload = {
        "env_alias": clean_interface.get("env_alias"),
        "components": schema["components"],
        "task_head": clean_interface.get("eureka_task_description", "")[:1000],
    }
    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    schema["schema_version"] = str(schema.get("schema_version") or f"eg_rsa_reward_schema_v1_{schema_hash}")
    if not schema["schema_version"].startswith("eg_rsa_reward_schema"):
        schema["schema_version"] = f"eg_rsa_reward_schema_v1_{schema_hash}"

    return schema
'''

if old not in s:
    raise SystemExit("Could not find old normalize_schema block. Patch manually.")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
PY

echo "[3/7] patch agents.py: strip import lines and support candidate lessons..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/agents.py")
s = p.read_text(encoding="utf-8")

insert_after = '''def _contains_private_term(text: str) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in PRIVATE_TERMS)
'''

helper = '''

def _strip_import_lines(code: str) -> str:
    """
    Programmatic fallback: remove import lines from generated reward code.

    The validator still forbids imports. This fallback avoids wasting repair
    attempts when the LLM adds harmless imports like import math or import numpy.
    """
    cleaned = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        cleaned.append(line)
    return "\\n".join(cleaned).strip() + "\\n"
'''

if "_strip_import_lines" not in s:
    s = s.replace(insert_after, insert_after + helper)

s = s.replace(
'''    return code.strip(), rationale
''',
'''    return _strip_import_lines(code.strip()), rationale
''',
)

old_extract = '''    def extract(
        self,
        evidence: dict[str, Any],
        reflection_report: str,
        scope: str,
        env_alias: str,
        generation: int,
        log_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user = (
            f"Scope: {scope}\\n"
            f"Env alias: {env_alias}\\n"
            f"Generation: {generation}\\n\\n"
            "Structured evidence:\\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\\n\\n"
            "Reflection report:\\n"
            f"{reflection_report}\\n\\n"
            "Return JSON array of lessons. Do not include code blocks."
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "LessonExtractorAgent", "scope": scope})
        lessons = _extract_json_array(response)
        return lessons, budget
'''

new_extract = '''    def extract(
        self,
        evidence: dict[str, Any],
        reflection_report: str,
        scope: str,
        env_alias: str,
        generation: int,
        log_dir: Path,
        candidate_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user = (
            f"Scope: {scope}\\n"
            f"Env alias: {env_alias}\\n"
            f"Generation: {generation}\\n"
            f"Candidate ID: {candidate_id or 'N/A'}\\n\\n"
            "Structured evidence:\\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\\n\\n"
            "Reflection report:\\n"
            f"{reflection_report}\\n\\n"
            "Return JSON array of lessons. Do not include code blocks."
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(
            log_dir,
            self.system_prompt,
            user,
            response,
            {"agent": "LessonExtractorAgent", "scope": scope, "candidate_id": candidate_id},
        )
        lessons = _extract_json_array(response)
        return lessons, budget
'''

if old_extract not in s:
    raise SystemExit("Could not find LessonExtractorAgent.extract block. Patch manually.")

s = s.replace(old_extract, new_extract)
p.write_text(s, encoding="utf-8")
PY

echo "[4/7] patch lessons.py: candidate evidence and LTM current-env exclusion..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/lessons.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
'''    ltm_lessons = [
        x for x in ltm_lessons
        if x.get("reuse_policy") in ("global", "similar_env", None)
    ][-ltm_lesson_top_k:]
''',
'''    # Do not retrieve current-environment lessons from LTM. Otherwise the
    # same generation's environment lessons can immediately re-enter as
    # "cross-environment" memory and pollute the next prompt.
    ltm_lessons = [
        x for x in ltm_lessons
        if x.get("reuse_policy") in ("global", "similar_env", None)
        and x.get("env_alias") != env_alias
    ][-ltm_lesson_top_k:]
''',
)

append_func = r'''

def pack_candidate_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Compact evidence for candidate-level lesson extraction."""
    return {
        "generation": record.get("generation"),
        "candidate_id": record.get("candidate_id"),
        "parent_ids": record.get("parent_ids", []),
        "status": record.get("status"),
        "selection_score": record.get("selection_score"),
        "private_eval_return": record.get("hidden_eval_return"),
        "generated_return": record.get("train_mean_return"),
        "generated_minus_private": float(record.get("train_mean_return", 0.0)) - float(record.get("hidden_eval_return", 0.0)),
        "repair_attempts": record.get("repair_attempts", 0),
        "repair_success": record.get("repair_success", False),
        "validation_errors": record.get("validation_errors", []),
        "diagnostics": record.get("diagnostics", {}),
        "reward_code_head": str(record.get("reward_code", ""))[:3500],
        "llm_rationale": str(record.get("llm_rationale", ""))[:1500],
    }
'''

if "def pack_candidate_evidence" not in s:
    s += append_func

p.write_text(s, encoding="utf-8")
PY

echo "[5/7] patch orchestrator.py: extract candidate-level lessons..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/orchestrator.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
'''    pack_generation_evidence,
    read_jsonl,
    retrieve_memory_context,
)''',
'''    pack_candidate_evidence,
    pack_generation_evidence,
    read_jsonl,
    retrieve_memory_context,
)''',
)

old = '''                self.memory.append(rec)
                as_dict = rec.__dict__
                generation_records.append(as_dict)

                generation_best = max(generation_best, selection_score)
'''

new = '''                as_dict = rec.__dict__

                # Candidate-level lesson extraction.
                # This gives STM a real lesson layer instead of only storing raw candidate records.
                lesson_ids: list[str] = []
                try:
                    candidate_evidence = pack_candidate_evidence(as_dict)
                    candidate_lessons_raw, candidate_lesson_budget = self.lesson_extractor.extract(
                        evidence=candidate_evidence,
                        reflection_report=rationale,
                        scope="candidate",
                        env_alias=clean_interface["env_alias"],
                        generation=g,
                        candidate_id=cid,
                        log_dir=candidate_llm_dir / "lesson_extractor_candidate",
                    )
                    candidate_lessons = [
                        normalize_lesson(
                            x,
                            scope="candidate",
                            env_alias=clean_interface["env_alias"],
                            generation=g,
                            candidate_id=cid,
                        )
                        for x in candidate_lessons_raw
                    ]
                    append_jsonl(self.cfg.candidate_lessons_path, candidate_lessons)
                    lesson_ids = [str(x.get("lesson_id")) for x in candidate_lessons]
                except Exception as e:
                    candidate_lessons = [
                        normalize_lesson(
                            {
                                "lesson_type": "extractor_error",
                                "condition": "Candidate lesson extraction failed.",
                                "observation": str(e),
                                "explanation": "Candidate lesson extraction raised an exception.",
                                "recommendation": "Inspect candidate lesson extractor prompt/response.",
                                "confidence": 0.2,
                                "reuse_policy": "same_env",
                            },
                            scope="candidate",
                            env_alias=clean_interface["env_alias"],
                            generation=g,
                            candidate_id=cid,
                        )
                    ]
                    append_jsonl(self.cfg.candidate_lessons_path, candidate_lessons)
                    lesson_ids = [str(x.get("lesson_id")) for x in candidate_lessons]

                rec.lesson_ids = lesson_ids
                self.memory.append(rec)
                as_dict = rec.__dict__
                generation_records.append(as_dict)

                generation_best = max(generation_best, selection_score)
'''

if old not in s:
    raise SystemExit("Could not find memory append block. Patch manually.")

s = s.replace(old, new)

# Stop writing current env lessons into LTM during each generation.
s = s.replace(
'''            append_jsonl(self.cfg.ltm_lessons_path, [
                normalize_lesson(x, scope="cross_environment", env_alias=clean_interface["env_alias"], generation=g)
                for x in env_lessons_raw
                if str(x.get("reuse_policy", "")).lower() in ("global", "cross_environment", "similar_env")
            ])
''',
'''            # Quality-gate v1: do not push current-run lessons into LTM during the same run.
            # LTM should be updated by a separate cross-run promotion step, not inside each generation.
'''
)

p.write_text(s, encoding="utf-8")
PY

echo "[6/7] patch prompts and add quality checker..."
cat > mvp/prompts/schema_planner_system.txt <<'TXT'
You are a schema-planning agent for reinforcement-learning reward search.

Use the Eureka-style task description, Eureka-processed step.py, and environment understanding to propose:
1. an environment-aware reward schema;
2. a search plan for reward generation.

You may use task semantics from task_description.txt and step.py.
Do not reconstruct, infer, or imitate any hidden implementation of the official reward or fitness evaluator.

The reward function can only use:
obs, action, next_obs, done, info.

Schema constraints:
- Return 4 to 6 required components.
- Avoid overlapping terminal components.
- Use exactly one terminal-outcome component if terminal shaping is needed.
- Do not simultaneously require landing_bonus, crash_penalty, and terminal.
- Keep components interpretable and balanced.
- Prefer components with comparable expected magnitudes.

Return JSON only with keys:
{
  "reward_schema": {
    "components": [
      {"id": "...", "description": "...", "direction": "maximize", "required": true}
    ],
    "reward_abs_bound": 1000.0
  },
  "search_plan": "markdown string"
}
TXT

cat > mvp/prompts/reward_coder_system.txt <<'TXT'
You are a reward engineer writing effective reward functions for reinforcement learning.

Use:
- Eureka task_description.txt
- Eureka-processed step.py
- environment understanding
- reward schema
- search plan
- feedback context
- retrieved memory lessons
- parent reward code

You may reason freely about observation/action semantics from the task files.

Hard constraints for generated code:
- Define compute_reward(obs, action, next_obs, done, info).
- Return float(total_reward), components_dict.
- Include all required schema component IDs.
- Do not use the environment reward returned by env.step.
- Do not reconstruct, infer, or imitate the hidden implementation of the official reward or fitness evaluator.
- Do not use private objective/evaluation implementation details.
- Do not write import statements.
- np and math are already available; do not import them.

Terminal logic guidance:
- Use one terminal-outcome component unless the schema explicitly says otherwise.
- Avoid double-counting done outcomes with multiple terminal components.

Output a Python code block containing compute_reward, followed by RATIONALE:<short explanation>.
TXT

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

For candidate scope:
- Focus on candidate-specific causes of success/failure.
- Mention component imbalance, inactive components, generated/private gap, action collapse, repair/validation problems, or useful mutations.

For environment scope:
- Focus on recurring patterns across candidates in this generation.

Do not include code blocks.
TXT

mkdir -p scripts

cat > scripts/check_run_quality.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--max-schema-components", type=int, default=6)
    parser.add_argument("--max-input-tokens", type=int, default=18000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    errors = []
    warnings = []

    schema_path = run_dir / "reward_schema.txt"
    if not schema_path.exists():
        errors.append("missing reward_schema.txt")
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        comps = schema.get("components", [])
        if len(comps) > args.max_schema_components:
            errors.append(f"schema has too many components: {len(comps)} > {args.max_schema_components}")
        terminal_like = [
            c.get("id", "")
            for c in comps
            if any(t in str(c.get("id", "")).lower() for t in ("terminal", "landing", "crash", "success", "failure", "done"))
        ]
        if len(terminal_like) > 1:
            errors.append(f"multiple terminal-like components: {terminal_like}")

    memory_rows = read_jsonl(run_dir / "memory.jsonl")
    if not memory_rows:
        errors.append("missing or empty memory.jsonl")

    candidate_lessons = read_jsonl(run_dir / "candidate_lessons.jsonl")
    if memory_rows and not candidate_lessons:
        errors.append("missing candidate_lessons.jsonl or no candidate lessons generated")

    env_lessons = read_jsonl(run_dir / "env_lessons.jsonl")
    if memory_rows and not env_lessons:
        errors.append("missing env_lessons.jsonl or no environment lessons generated")

    invalid_import = []
    for r in memory_rows:
        errs = " ".join(map(str, r.get("validation_errors", [])))
        if "Import" in errs or "import" in errs:
            invalid_import.append(r.get("candidate_id"))
    if invalid_import:
        warnings.append(f"candidates still failed due to imports: {invalid_import}")

    for budget_path in run_dir.glob("llm/**/budget.json"):
        try:
            b = json.loads(budget_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        n = int(b.get("estimated_input_tokens", 0))
        if n > args.max_input_tokens:
            warnings.append(f"large prompt: {budget_path} estimated_input_tokens={n}")

    # Check LTM/env duplication in memory_context.
    for mem_ctx in run_dir.glob("artifacts/generation_*/memory_context.txt"):
        text = mem_ctx.read_text(encoding="utf-8", errors="replace")
        if "Relevant cross-environment lessons:" in text:
            cross_block = text.split("Relevant cross-environment lessons:", 1)[1]
            if "env_alias" in cross_block:
                warnings.append(f"memory_context may include raw env_alias in cross lessons: {mem_ctx}")

    print("=== EG-RSA RUN QUALITY CHECK ===")
    print(f"run_dir: {run_dir}")
    print(f"num_memory_rows: {len(memory_rows)}")
    print(f"num_candidate_lessons: {len(candidate_lessons)}")
    print(f"num_env_lessons: {len(env_lessons)}")

    if warnings:
        print("\\nWARNINGS:")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\\nERRORS:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("\\nOK: quality checks passed.")


if __name__ == "__main__":
    main()
PY

chmod +x scripts/check_run_quality.py

echo "[7/7] syntax checks..."
python -m py_compile mvp/*.py scripts/check_run_quality.py scripts/check_eureka_step_input.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Next:"
echo "  python scripts/check_eureka_step_input.py --env-id LunarLander-v3"
echo "  rm -rf runs/eg_rsa_lunar_deepseek_smoke"
echo "  bash scripts/run_eg_rsa_lunar_smoke.sh"
echo "  python scripts/check_run_quality.py runs/eg_rsa_lunar_deepseek_smoke"
