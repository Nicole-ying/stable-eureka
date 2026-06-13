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
