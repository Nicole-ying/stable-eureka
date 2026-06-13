#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when this file is executed as:
#   python scripts/check_eureka_step_input.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.env_sanitizer import infer_clean_env_interface
from mvp.task_specs import make_env_alias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="LunarLander-v3")
    args = parser.parse_args()

    env_alias = make_env_alias(args.env_id)
    interface = infer_clean_env_interface(args.env_id, env_alias)

    step_path = Path(interface["source_files"]["step"])
    raw_step = step_path.read_text(encoding="utf-8", errors="replace")
    prompt_step = interface["eureka_step_code"]

    if raw_step != prompt_step:
        raise SystemExit("FAILED: prompt step.py is not exactly the Eureka-processed step.py on disk.")

    forbidden_defs = [
        "def compute_reward",
        "def compute_fitness_score",
    ]
    leaked_defs = [x for x in forbidden_defs if x in prompt_step]
    if leaked_defs:
        raise SystemExit(f"FAILED: step.py contains private implementation definitions: {leaked_defs}")

    print("OK: LLM receives exact Eureka-processed step.py.")
    print("OK: step.py contains no compute_reward / compute_fitness_score implementation definitions.")
    print(f"step_path={step_path}")
    print(f"step_lines={len(prompt_step.splitlines())}")


if __name__ == "__main__":
    main()
