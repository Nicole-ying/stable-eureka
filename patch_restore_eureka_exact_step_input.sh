#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] check repo root..."
test -d mvp || { echo "ERROR: run this script at repo root"; exit 1; }

backup_dir="backup_before_restore_eureka_exact_step_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/env_sanitizer.py \
  mvp/prompts/env_understanding_system.txt \
  mvp/prompts/schema_planner_system.txt \
  mvp/prompts/reward_coder_system.txt \
  docs/EG_RSA_FRAMEWORK.md \
  scripts/check_redacted_step.py \
  scripts/check_eureka_step_input.py
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/6] restore env_sanitizer.py to Eureka-exact step input..."
cat > mvp/env_sanitizer.py <<'PY'
from __future__ import annotations

from pathlib import Path
from typing import Any


ENV_ID_TO_EUREKA_DIRS: dict[str, list[str]] = {
    "LunarLander-v3": ["lunar_lander"],
    "LunarLander-v2": ["lunar_lander"],
    "BipedalWalker-v3": ["bipedal_walker"],
    "CartPole-v1": ["cartpole", "cart_pole"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_env_dirs(env_id: str) -> list[Path]:
    root = _repo_root()
    out: list[Path] = []

    for name in ENV_ID_TO_EUREKA_DIRS.get(env_id, []):
        out.append(root / "envs" / name)

    stem = env_id.split("-")[0]
    candidates = {
        stem,
        stem.lower(),
        stem.replace("LunarLander", "lunar_lander"),
        stem.replace("BipedalWalker", "bipedal_walker"),
        stem.replace("CartPole", "cartpole"),
    }
    for c in candidates:
        out.append(root / "envs" / c)

    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _find_eureka_file(env_id: str, filename: str) -> Path:
    for d in _candidate_env_dirs(env_id):
        p = d / filename
        if p.exists():
            return p

    searched = "\n".join(str(d / filename) for d in _candidate_env_dirs(env_id))
    raise FileNotFoundError(
        f"Eureka-style file not found for env_id={env_id}: {filename}\n"
        f"Searched:\n{searched}\n\n"
        "EG-RSA requires Eureka-style envs/<task>/task_description.txt and envs/<task>/step.py. "
        "It does not synthesize extra observation/action range tables."
    )


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _assert_eureka_step_has_no_objective_implementation(step_code: str) -> None:
    """
    Eureka step.py may contain hook calls such as:
      self.compute_reward(...)
      self.compute_fitness_score(...)

    That is acceptable for Eureka-aligned input.

    What must not be present is the implementation body of official reward
    or hidden evaluator functions inside the step.py file.
    """
    forbidden_defs = [
        "def compute_reward",
        "def compute_fitness_score",
        "class Reward",
    ]
    lower = step_code.lower()
    leaked = [x for x in forbidden_defs if x.lower() in lower]
    if leaked:
        raise ValueError(
            "Eureka step.py appears to contain private objective/evaluator implementation definitions: "
            f"{leaked}"
        )


def infer_clean_env_interface(env_id: str, env_alias: str) -> dict[str, Any]:
    """
    Final EG-RSA task input.

    We follow Eureka exactly:
      1. task_description.txt
      2. Eureka-processed step.py

    We do not add a Gym-space-derived observation/action range table.
    We also do not additionally redact hook lines that are already part of
    Eureka's provided step.py. The hidden reward/evaluator implementation
    itself must not be present.
    """
    task_path = _find_eureka_file(env_id, "task_description.txt")
    step_path = _find_eureka_file(env_id, "step.py")

    step_code = _read_file(step_path)
    _assert_eureka_step_has_no_objective_implementation(step_code)

    return {
        "interface_mode": "eureka_exact_step",
        "env_alias": env_alias,
        "eureka_task_description": _read_file(task_path),
        "eureka_step_code": step_code,
        "source_files": {
            "task_description": str(task_path),
            "step": str(step_path),
        },
        "step_policy": {
            "mode": "eureka_exact",
            "description": (
                "The LLM receives the same Eureka-processed step.py. "
                "Hook calls to compute_reward/compute_fitness_score may appear, "
                "but their implementations are not provided."
            ),
        },
        "reward_function_contract": {
            "signature": "compute_reward(obs, action, next_obs, done, info)",
            "visible_inputs": ["obs", "action", "next_obs", "done", "info"],
            "return": "float(total_reward), components_dict",
        },
        "input_boundary": {
            "allowed": [
                "task_description.txt",
                "Eureka-processed step.py",
                "LLM reasoning over observation/action semantics from public task files",
                "environment understanding generated from task files",
                "reward schema and search plan generated from task files",
                "parent reward code",
                "training feedback",
                "STM/MTM/LTM lessons retrieved from memory",
            ],
            "generated_code_forbidden": [
                "environment reward returned by env.step",
                "official reward implementation",
                "fitness/evaluation implementation",
                "hidden evaluator implementation",
                "expert reward template",
            ],
        },
    }
PY

echo "[3/6] update prompt files to say Eureka-processed step.py, not redacted step.py..."
cat > mvp/prompts/env_understanding_system.txt <<'TXT'
You are an environment-understanding agent for reinforcement-learning reward design.

Input boundary:
- You are given Eureka-style task_description.txt and the Eureka-processed step.py.
- You may reason freely about task semantics, observation meanings, action meanings, state construction, and termination conditions from these task files.
- The step.py may contain hook calls such as compute_reward or compute_fitness_score because Eureka also provides step.py this way.
- Do not reconstruct, infer, or imitate the hidden implementation of the official reward or fitness evaluator.

Important reward-function input boundary:
- The generated reward function will only receive: obs, action, next_obs, done, info.
- Local variables inside step.py are not direct reward-function inputs unless they can be inferred from obs/action/next_obs/done/info.
- If step.py contains internal engine variables, describe how they relate to action, but do not list them as direct reward inputs.

Output:
- A concise markdown environment understanding report.
- Then a JSON object with keys:
  task_goal, observations, actions, termination, reward_function_visible_inputs, inferable_public_quantities, risks.
TXT

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
- Prefer 4 to 6 required components.
- Avoid overlapping terminal components.
- Use exactly one terminal-outcome component if terminal shaping is needed.
- Avoid separate landing_bonus, crash_penalty, and terminal components that all activate on done.
- Keep components interpretable and balanced.

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
- Do not import packages.
- Use only Python builtins, math, and numpy as np.

Terminal logic guidance:
- Use one terminal-outcome component unless the schema explicitly says otherwise.
- Avoid double-counting done outcomes with multiple terminal components.

Output a Python code block containing compute_reward, followed by RATIONALE:<short explanation>.
TXT

echo "[4/6] remove old redaction checker and add Eureka-exact input checker..."
rm -f scripts/check_redacted_step.py
mkdir -p scripts

cat > scripts/check_eureka_step_input.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

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
PY

chmod +x scripts/check_eureka_step_input.py

echo "[5/6] update docs..."
python - <<'PY'
from pathlib import Path

p = Path("docs/EG_RSA_FRAMEWORK.md")
if not p.exists():
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# EG-RSA Final Framework\n\n", encoding="utf-8")

s = p.read_text(encoding="utf-8")

s = s.replace(
    "envs/<task>/step.py after private objective/evaluation redaction",
    "envs/<task>/step.py, exactly as provided by Eureka"
)
s = s.replace(
    "redacted step.py",
    "Eureka-processed step.py"
)
s = s.replace(
    "redacted Eureka step.py",
    "Eureka-processed step.py"
)

if "## Eureka Step Policy" not in s:
    s += """

## Eureka Step Policy

EG-RSA uses the same task-code input policy as Eureka.

The LLM receives:

1. task_description.txt
2. step.py exactly as provided in the Eureka-style envs/<task>/ directory

The step.py may contain hook calls such as compute_reward or compute_fitness_score. These hook calls are part of the Eureka-provided environment code and are allowed in the input.

The hidden implementations of the official reward and fitness evaluator are not provided. Generated reward code must not reconstruct or imitate those hidden implementations.
"""

p.write_text(s, encoding="utf-8")
PY

echo "[6/6] syntax and input checks..."
python -m py_compile mvp/*.py scripts/check_eureka_step_input.py
python scripts/check_eureka_step_input.py --env-id LunarLander-v3

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Meaning:"
echo "  - We now follow Eureka exactly for step.py input."
echo "  - No custom redaction is applied."
echo "  - We only check that step.py does not contain objective/evaluator implementation definitions."
