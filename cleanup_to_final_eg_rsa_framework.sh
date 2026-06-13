#!/usr/bin/env bash
set -euo pipefail

echo "[1/12] check repo root..."
test -d mvp || { echo "ERROR: run this script at repo root"; exit 1; }

backup_dir="backup_before_final_eg_rsa_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

echo "[2/12] backup current important files..."
for f in \
  mvp/config.py \
  mvp/env_sanitizer.py \
  mvp/rl_worker.py \
  mvp/task_specs.py \
  mvp/orchestrator.py \
  mvp/agents.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/reward_schema.py \
  mvp/lessons.py \
  mvp/llm_logging.py
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

if [ -d runs ]; then
  mkdir -p "$backup_dir/runs_snapshot"
  find runs -maxdepth 2 -type f \( \
    -name "report.md" \
    -o -name "memory.csv" \
    -o -name "memory.jsonl" \
    -o -name "env_lessons.jsonl" \
    -o -name "env_memory.md" \
  \) -print -exec cp --parents {} "$backup_dir/runs_snapshot" \; || true
fi

echo "[3/12] remove obsolete old-framework modules..."
rm -f mvp/semantic_audit.py
rm -f mvp/leak_audit.py
rm -f mvp/clean_feedback.py

echo "[4/12] rewrite task_specs.py..."
cat > mvp/task_specs.py <<'PY'
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivateTaskSpec:
    """
    Runtime-only task identity.

    EG-RSA follows Eureka-style task input:
      - task_description.txt
      - step.py

    The real env_id is used only by runtime to create Gym environments
    and evaluate policies. It is not used to synthesize extra task
    semantics beyond the Eureka task files.
    """
    env_id: str
    hidden_eval_source: str = "env_reward"
    benchmark_id: str | None = None


PRIVATE_TASK_SPECS: dict[str, PrivateTaskSpec] = {
    "LunarLander-v3": PrivateTaskSpec(env_id="LunarLander-v3", benchmark_id="LunarLander-v3"),
    "LunarLander-v2": PrivateTaskSpec(env_id="LunarLander-v2", benchmark_id="LunarLander-v2"),
    "BipedalWalker-v3": PrivateTaskSpec(env_id="BipedalWalker-v3", benchmark_id="BipedalWalker-v3"),
    "CartPole-v1": PrivateTaskSpec(env_id="CartPole-v1", benchmark_id="CartPole-v1"),
}


def make_env_alias(env_id: str) -> str:
    digest = hashlib.sha1(env_id.encode("utf-8")).hexdigest()[:8]
    return f"Env-{digest}"


def get_private_task_spec(env_id: str) -> PrivateTaskSpec:
    return PRIVATE_TASK_SPECS.get(env_id, PrivateTaskSpec(env_id=env_id, benchmark_id=env_id))
PY

echo "[5/12] rewrite env_sanitizer.py to final Eureka-only input..."
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
        "Final EG-RSA framework requires envs/<task>/task_description.txt and envs/<task>/step.py. "
        "It does not synthesize extra observation/action range tables."
    )


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def infer_clean_env_interface(env_id: str, env_alias: str) -> dict[str, Any]:
    """
    Final EG-RSA task input.

    The LLM receives exactly Eureka-style task context:
      1. task_description.txt
      2. step.py

    The framework may add memory, lessons, feedback, schema, and parent reward code
    during iteration. It does not add a separate Gym-space-derived observation/action
    range table.
    """
    task_path = _find_eureka_file(env_id, "task_description.txt")
    step_path = _find_eureka_file(env_id, "step.py")

    return {
        "interface_mode": "eureka_clean",
        "env_alias": env_alias,
        "eureka_task_description": _read_file(task_path),
        "eureka_step_code": _read_file(step_path),
        "source_files": {
            "task_description": str(task_path),
            "step": str(step_path),
        },
        "reward_function_contract": {
            "signature": "compute_reward(obs, action, next_obs, done, info)",
            "visible_inputs": ["obs", "action", "next_obs", "done", "info"],
            "return": "float(total_reward), components_dict",
        },
        "input_boundary": {
            "allowed": [
                "task_description.txt",
                "step.py",
                "LLM reasoning over observation/action semantics",
                "environment understanding generated from task files",
                "reward schema and search plan generated from task files",
                "parent reward code",
                "training feedback",
                "STM/MTM/LTM lessons retrieved from memory",
            ],
            "generated_code_forbidden": [
                "env_reward",
                "official reward formula",
                "fitness_score",
                "compute_fitness_score",
                "hidden evaluator implementation",
                "expert reward template",
            ],
        },
    }
PY

echo "[6/12] rewrite rl_worker.py to remove anonymous branch..."
cat > mvp/rl_worker.py <<'PY'
from __future__ import annotations

import ast
import math
import types
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from .config import RLConfig


RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, bool, dict], tuple[float, dict]]


FORBIDDEN_NAMES = {
    "env_reward",
    "fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
    "_hidden_env_reward",
    "compute_fitness_score",
}


def compile_reward_function(reward_code: str) -> RewardFn:
    raw_lower = reward_code.lower()
    for name in FORBIDDEN_NAMES:
        if name.lower() in raw_lower:
            raise ValueError(f"forbidden token appears in reward code: {name}")

    tree = ast.parse(reward_code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.ClassDef, ast.Lambda)):
            raise ValueError(f"unsupported syntax in reward code: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden name in reward code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden attribute in reward code: {node.attr}")

    module = types.ModuleType("reward_module")
    module.__dict__["np"] = np
    module.__dict__["math"] = math
    exec(compile(tree, filename="<reward_code>", mode="exec"), module.__dict__)

    if "compute_reward" not in module.__dict__:
        raise ValueError("reward code must define compute_reward(obs, action, next_obs, done, info)")
    return module.__dict__["compute_reward"]


class RewardFunctionWrapper(gym.Wrapper):
    """
    Final EG-RSA wrapper.

    Policy sees the original Gym observation.
    Reward function sees public transition inputs:
      obs, action, next_obs, done, info

    It never receives env_reward. The original env reward is stored only as
    private evaluation signal in safe_info["_hidden_env_reward"].
    """

    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        next_obs, hidden_env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        out = self.reward_fn(
            self._prev_obs,
            action,
            next_obs,
            done,
            info,
        )
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError("compute_reward must return (reward, components)")

        reward, components = out
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("generated reward is not finite")
        if not isinstance(components, dict):
            raise ValueError("reward components must be a dict")

        safe_info = dict(info)
        safe_info["_hidden_env_reward"] = float(hidden_env_reward)
        safe_info["_reward_components"] = {str(k): float(v) for k, v in components.items()}

        self._prev_obs = next_obs
        return next_obs, reward, terminated, truncated, safe_info


class RLWorker:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg

    def train_and_eval(self, reward_code: str, ckpt_path: Path) -> dict[str, object]:
        reward_fn = compile_reward_function(reward_code)

        train_env = RewardFunctionWrapper(gym.make(self.cfg.env_id), reward_fn)
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=self.cfg.learning_rate,
            gamma=self.cfg.gamma,
        )
        model.learn(total_timesteps=self.cfg.total_timesteps)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt_path))
        train_env.close()

        eval_env = RewardFunctionWrapper(gym.make(self.cfg.env_id), reward_fn)

        generated_returns = []
        hidden_returns = []
        episode_lengths = []
        component_returns: dict[str, list[float]] = {}
        action_values = []

        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            gen_ret = 0.0
            hid_ret = 0.0
            ep_len = 0
            comp_sum: dict[str, float] = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_values.append(np.asarray(action, dtype=float).reshape(-1))

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)

                gen_ret += float(reward)
                hid_ret += float(info.get("_hidden_env_reward", 0.0))
                ep_len += 1

                for k, v in info.get("_reward_components", {}).items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + float(v)

            generated_returns.append(gen_ret)
            hidden_returns.append(hid_ret)
            episode_lengths.append(ep_len)

            for k, v in comp_sum.items():
                component_returns.setdefault(k, []).append(v)

        eval_env.close()

        action_arr = np.concatenate(action_values) if action_values else np.zeros((1,), dtype=float)
        component_mean = {k: float(np.mean(v)) for k, v in component_returns.items()}

        return {
            "eval_generated_return": float(np.mean(generated_returns)),
            "eval_hidden_return": float(np.mean(hidden_returns)),
            "eval_episode_length": float(np.mean(episode_lengths)),
            "component_returns": component_mean,
            "diagnostics": {
                "generated_private_gap": float(np.mean(generated_returns) - np.mean(hidden_returns)),
                "action_mean": float(np.mean(action_arr)),
                "action_std": float(np.std(action_arr)),
                "episode_length_mean": float(np.mean(episode_lengths)),
                "component_returns": component_mean,
            },
        }

    def render_rollout_video(self, ckpt_path: Path, video_path: Path) -> Path:
        model = PPO.load(str(ckpt_path))
        env = gym.make(self.cfg.env_id, render_mode="rgb_array")
        obs, _ = env.reset()
        done = False
        frames = []

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

        video_path.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            imageio.mimsave(video_path, frames, fps=30)
        env.close()
        return video_path
PY

echo "[7/12] patch config.py: remove interface_mode if present..."
python - <<'PY'
from pathlib import Path
import re

p = Path("mvp/config.py")
s = p.read_text(encoding="utf-8")

s = re.sub(
    r'\n    # eureka_clean:[\s\S]*?interface_mode: str = "eureka_clean"\n',
    "\n",
    s,
)

s = s.replace('    interface_mode: str = "eureka_clean"\n\n', '')

p.write_text(s, encoding="utf-8")
PY

echo "[8/12] patch orchestrator.py: remove interface_mode argument..."
python - <<'PY'
from pathlib import Path
import re

p = Path("mvp/orchestrator.py")
s = p.read_text(encoding="utf-8")

old_block = '''        interface_mode = getattr(self.cfg.rl, "interface_mode", "eureka_clean")

        clean_interface = infer_clean_env_interface(
            private_task.env_id,
            env_alias,
            interface_mode=interface_mode,
        )
'''
new_block = '''        clean_interface = infer_clean_env_interface(
            private_task.env_id,
            env_alias,
        )
'''

if old_block in s:
    s = s.replace(old_block, new_block)
else:
    s = re.sub(
        r'\n        interface_mode = getattr\(self\.cfg\.rl, "interface_mode", "eureka_clean"\)\n\n        clean_interface = infer_clean_env_interface\(\n            private_task\.env_id,\n            env_alias,\n            interface_mode=interface_mode,\n        \)\n',
        "\n        clean_interface = infer_clean_env_interface(\n            private_task.env_id,\n            env_alias,\n        )\n",
        s,
    )

p.write_text(s, encoding="utf-8")
PY

echo "[9/12] replace old configs with final EG-RSA configs..."
mkdir -p mvp/configs
mkdir -p "$backup_dir/mvp"
cp -r mvp/configs "$backup_dir/mvp/configs_before_cleanup" 2>/dev/null || true

rm -f mvp/configs/*clean*.yaml
rm -f mvp/configs/*anonymous*.yaml

cat > mvp/configs/eg_rsa_lunar_deepseek_smoke.yaml <<'YAML'
model:
  provider: deepseek
  llm_model: deepseek-v4-flash
  vlm_model: deepseek-v4-flash
  deepseek_base_url: https://api.deepseek.com
  deepseek_api_key_env: DEEPSEEK_API_KEY
  deepseek_thinking: disabled
  temperature: 0.7
  max_tokens: 2500

rl:
  env_id: LunarLander-v3
  total_timesteps: 1000
  eval_episodes: 1
  learning_rate: 0.0003
  gamma: 0.99

evolution:
  generations: 1
  population_size: 1
  elite_size: 1
  reflection_top_k: 1

memory:
  candidate_lesson_top_k: 3
  env_lesson_top_k: 5
  ltm_lesson_top_k: 3
  parent_code_top_k: 1
  parent_code_max_chars: 8000
  feedback_max_chars: 8000
  memory_context_max_chars: 8000

workspace: runs/eg_rsa_lunar_deepseek_smoke
seed: 0
YAML

cat > mvp/configs/eg_rsa_lunar_deepseek_seed0.yaml <<'YAML'
model:
  provider: deepseek
  llm_model: deepseek-v4-flash
  vlm_model: deepseek-v4-flash
  deepseek_base_url: https://api.deepseek.com
  deepseek_api_key_env: DEEPSEEK_API_KEY
  deepseek_thinking: disabled
  temperature: 0.7
  max_tokens: 2500

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

memory:
  candidate_lesson_top_k: 5
  env_lesson_top_k: 8
  ltm_lesson_top_k: 5
  parent_code_top_k: 2
  parent_code_max_chars: 12000
  feedback_max_chars: 12000
  memory_context_max_chars: 12000

workspace: runs/eg_rsa_lunar_deepseek_seed0_g2p2_t30k
seed: 0
YAML

echo "[10/12] cleanup runs: keep only renamed latest EG-RSA run and ltm..."
mkdir -p "$backup_dir/full_runs_backup"

if [ -d runs ]; then
  find runs -maxdepth 2 -type f \( \
    -name "report.md" \
    -o -name "memory.csv" \
    -o -name "memory.jsonl" \
    -o -name "env_lessons.jsonl" \
    -o -name "env_memory.md" \
  \) -print -exec cp --parents {} "$backup_dir/full_runs_backup" \; || true

  if [ -d runs/clean_lunar_lander_deepseek_seed0_g2p2_t30k ]; then
    rm -rf runs/eg_rsa_lunar_deepseek_seed0_g2p2_t30k
    mv runs/clean_lunar_lander_deepseek_seed0_g2p2_t30k runs/eg_rsa_lunar_deepseek_seed0_g2p2_t30k
  fi

  find runs -maxdepth 1 -type d -name "clean_*" -exec rm -rf {} +
  find runs -maxdepth 1 -type d -name "anonymous_*" -exec rm -rf {} +
  find runs -maxdepth 1 -type d -name "mvp*" -exec rm -rf {} +
fi

echo "[11/12] add final framework doc and run script..."
mkdir -p docs scripts

cat > docs/EG_RSA_FRAMEWORK.md <<'MD'
# EG-RSA Final Framework

## Goal

EG-RSA follows Eureka-style task input while adding reward-search structure, repair, diagnostics, reflection, and three-level memory.

## Input Boundary

The LLM receives the same task cognition source as Eureka:

1. envs/<task>/task_description.txt
2. envs/<task>/step.py

The framework does not synthesize an extra Gym-space-derived observation/action range table.

Allowed:

- LLM reasoning over observation/action semantics from the task files
- environment understanding generated from task files
- reward schema and search plan generated from task files
- parent reward code
- training feedback
- STM, MTM, and LTM lessons

Forbidden in generated reward code:

- env_reward
- fitness_score
- compute_fitness_score
- official reward formula
- hidden evaluator implementation
- expert reward template

## LLM Agents

1. EnvUnderstandingAgent
   - Input: task_description plus step.py
   - Output: artifacts/env_understanding.md and artifacts/env_understanding.json

2. SchemaPlannerAgent
   - Input: task files plus environment understanding
   - Output: reward_schema.txt and clean_plan.txt

3. RewardCoderAgent
   - Input: task files, environment understanding, schema, search plan, feedback, memory context, parent reward code
   - Output: reward_code.py, rationale, raw LLM response

4. RepairAgent
   - Trigger: validator failure
   - Output: repaired reward code

5. ReflectionAgent
   - Trigger: generation end
   - Input: structured evidence, environment memory, retrieved lessons
   - Output: reflection_report.md

6. LessonExtractorAgent
   - Trigger: generation end
   - Output: environment lessons and cross-environment lessons

## Memory

### STM: Candidate Memory

Files:

- memory.jsonl
- memory.csv
- candidate artifacts under artifacts/generation_*

Used for:

- parent reward selection
- candidate-level lesson retrieval

### MTM: Environment Memory

Files:

- env_lessons.jsonl
- env_memory.md

Updated after every generation.

Used in the next generation as environment-level memory context.

### LTM: Cross-Environment Memory

File:

- runs/ltm_lessons.jsonl

Updated when a lesson is marked reusable beyond the current environment.

Retrieved in later runs as cross-task memory.

## Artifacts

Every LLM call is logged:

- llm/<stage>/system.txt
- llm/<stage>/user.txt
- llm/<stage>/response.txt
- llm/<stage>/budget.json

budget.json stores character counts and estimated token counts.

## Runtime Flow

task_description.txt plus step.py
  -> EnvUnderstandingAgent
  -> SchemaPlannerAgent
  -> MemoryRetriever
  -> RewardCoderAgent
  -> Validator
  -> RepairAgent if needed
  -> RL training and private evaluation
  -> EvidencePacker
  -> ReflectionAgent
  -> LessonExtractorAgent
  -> STM, MTM, and LTM update
  -> next generation
MD

cat > scripts/run_eg_rsa_lunar_smoke.sh <<'BASH2'
#!/usr/bin/env bash
set -euo pipefail

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY first}"

rm -rf runs/eg_rsa_lunar_deepseek_smoke

python run_mvp.py \
  --config mvp/configs/eg_rsa_lunar_deepseek_smoke.yaml
BASH2

chmod +x scripts/run_eg_rsa_lunar_smoke.sh

echo "[12/12] remove old backup dirs and run syntax checks..."
find . -maxdepth 1 -type d -name "backup_before_*" ! -name "$backup_dir" -exec rm -rf {} +

python -m py_compile mvp/*.py

echo ""
echo "FINAL EG-RSA CLEANUP DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Remaining intended run dirs:"
find runs -maxdepth 1 -type d 2>/dev/null | sort || true
echo ""
echo "Next smoke command:"
echo "  bash scripts/run_eg_rsa_lunar_smoke.sh"
