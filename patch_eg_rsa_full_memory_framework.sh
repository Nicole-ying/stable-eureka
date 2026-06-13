#!/usr/bin/env bash
set -euo pipefail

echo "[1/12] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_eg_rsa_full_memory_framework_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/config.py \
  mvp/env_sanitizer.py \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/memory.py \
  mvp/rl_worker.py \
  mvp/orchestrator.py \
  mvp/exporters.py \
  mvp/models.py \
  mvp/prompts/reward_coder_system.txt \
  mvp/prompts/repair_system.txt \
  mvp/prompts/env_understanding_system.txt \
  mvp/prompts/schema_planner_system.txt \
  mvp/prompts/reflection_system.txt \
  mvp/prompts/lesson_extractor_system.txt
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/12] write config.py..."
cat > mvp/config.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    provider: str = "openai"  # openai | deepseek | ollama | mock

    llm_model: str = "gpt-4.1"
    vlm_model: str = "gpt-4.1-mini"

    openai_base_url: str | None = None
    openai_api_key_env: str = "OPENAI_API_KEY"

    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_thinking: str = "disabled"
    deepseek_reasoning_effort: str | None = None

    ollama_host: str = "http://localhost:11434"
    ollama_num_ctx: int | None = 16384

    temperature: float = 0.7
    max_tokens: int = 2500


@dataclass
class RLConfig:
    env_id: str = "LunarLander-v3"

    # eureka_clean:
    #   Main EG-RSA setting. The LLM sees Eureka-style task_description.txt + step.py.
    # anonymous_clean:
    #   Optional ablation. The reward function sees anonymized observations.
    interface_mode: str = "eureka_clean"

    total_timesteps: int = 30_000
    eval_episodes: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99


@dataclass
class EvolutionConfig:
    generations: int = 4
    population_size: int = 3
    elite_size: int = 1
    reflection_top_k: int = 2
    target_score: float | None = None
    max_stagnation_generations: int | None = None


@dataclass
class MemoryConfig:
    candidate_lesson_top_k: int = 5
    env_lesson_top_k: int = 8
    ltm_lesson_top_k: int = 5
    parent_code_top_k: int = 2
    parent_code_max_chars: int = 12000
    feedback_max_chars: int = 12000
    memory_context_max_chars: int = 12000


@dataclass
class MVPConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    workspace: Path = Path("runs/mvp")
    seed: int = 42

    @property
    def memory_path(self) -> Path:
        return self.workspace / "memory.jsonl"

    @property
    def candidate_lessons_path(self) -> Path:
        return self.workspace / "candidate_lessons.jsonl"

    @property
    def env_lessons_path(self) -> Path:
        return self.workspace / "env_lessons.jsonl"

    @property
    def env_memory_path(self) -> Path:
        return self.workspace / "env_memory.md"

    @property
    def ltm_lessons_path(self) -> Path:
        # Cross-run / cross-environment memory under runs/.
        return self.workspace.parent / "ltm_lessons.jsonl"

    @property
    def llm_dir(self) -> Path:
        return self.workspace / "llm"

    @property
    def artifacts_dir(self) -> Path:
        return self.workspace / "artifacts"

    @property
    def videos_dir(self) -> Path:
        return self.workspace / "videos"

    @property
    def checkpoints_dir(self) -> Path:
        return self.workspace / "checkpoints"


def _deep_update(target: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            target[k] = _deep_update(target[k], v)
        else:
            target[k] = v
    return target


def load_config(path: str | Path | None = None) -> MVPConfig:
    cfg = MVPConfig()
    if path is None:
        return cfg

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = {
        "model": cfg.model.__dict__.copy(),
        "rl": cfg.rl.__dict__.copy(),
        "evolution": cfg.evolution.__dict__.copy(),
        "memory": cfg.memory.__dict__.copy(),
        "workspace": str(cfg.workspace),
        "seed": cfg.seed,
    }
    merged = _deep_update(base, raw)

    return MVPConfig(
        model=ModelConfig(**merged["model"]),
        rl=RLConfig(**merged["rl"]),
        evolution=EvolutionConfig(**merged["evolution"]),
        memory=MemoryConfig(**merged.get("memory", {})),
        workspace=Path(merged["workspace"]),
        seed=int(merged["seed"]),
    )
PY

echo "[3/12] write env_sanitizer.py..."
cat > mvp/env_sanitizer.py <<'PY'
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
        "EG-RSA eureka_clean requires envs/<task>/task_description.txt and envs/<task>/step.py. "
        "The framework will not synthesize extra task-interface fields silently."
    )


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _anonymous_space(space: spaces.Space) -> dict[str, Any]:
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "dimension_semantics": "not_available",
        }
    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
            "dimension_semantics": "not_available",
        }
    return {
        "type": type(space).__name__,
        "dimension_semantics": "not_available",
    }


class ObservationAnonymizer:
    def __init__(self, observation_space: spaces.Space):
        self.observation_space = observation_space

        if isinstance(observation_space, spaces.Box):
            low = np.asarray(observation_space.low, dtype=np.float32)
            high = np.asarray(observation_space.high, dtype=np.float32)
            finite = np.isfinite(low) & np.isfinite(high) & (high > low)

            center = np.zeros_like(low, dtype=np.float32)
            scale = np.ones_like(low, dtype=np.float32)

            center[finite] = (low[finite] + high[finite]) / 2.0
            scale[finite] = (high[finite] - low[finite]) / 2.0
            scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)

            self.box_finite = finite
            self.center = center
            self.scale = scale
        else:
            self.box_finite = None
            self.center = None
            self.scale = None

    def transform(self, obs):
        arr = np.asarray(obs, dtype=np.float32)

        if isinstance(self.observation_space, spaces.Box):
            out = np.zeros_like(arr, dtype=np.float32)
            finite = self.box_finite
            out[finite] = (arr[finite] - self.center[finite]) / self.scale[finite]
            out[~finite] = np.tanh(arr[~finite] / 5.0)
            return np.clip(out, -5.0, 5.0).astype(np.float32)

        return np.tanh(arr / 5.0).astype(np.float32)


def infer_clean_env_interface(
    env_id: str,
    env_alias: str,
    interface_mode: str = "eureka_clean",
) -> dict[str, Any]:
    if interface_mode == "anonymous_clean":
        env = gym.make(env_id)
        try:
            return {
                "interface_mode": "anonymous_clean",
                "env_alias": env_alias,
                "observation_space": _anonymous_space(env.observation_space),
                "action_space": _anonymous_space(env.action_space),
                "reward_function_contract": {
                    "signature": "compute_reward(obs, action, next_obs, done, info)",
                    "visible_inputs": ["obs", "action", "next_obs", "done", "info"],
                    "return": "float(total_reward), components_dict",
                },
            }
        finally:
            env.close()

    if interface_mode != "eureka_clean":
        raise ValueError(f"Unsupported interface_mode={interface_mode}. Use eureka_clean or anonymous_clean.")

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
                "parent reward code",
                "training feedback",
                "lessons retrieved from memory",
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

echo "[4/12] write llm_logging.py..."
cat > mvp/llm_logging.py <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    # Conservative language-agnostic heuristic.
    return max(1, int(len(text) / 3.5))


def write_llm_call(
    log_dir: Path,
    system: str,
    user: str,
    response: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)

    system_path = log_dir / "system.txt"
    user_path = log_dir / "user.txt"
    response_path = log_dir / "response.txt"
    budget_path = log_dir / "budget.json"

    system_path.write_text(system, encoding="utf-8")
    user_path.write_text(user, encoding="utf-8")
    response_path.write_text(response, encoding="utf-8")

    budget = {
        "system_chars": len(system),
        "user_chars": len(user),
        "response_chars": len(response),
        "estimated_system_tokens": estimate_tokens(system),
        "estimated_user_tokens": estimate_tokens(user),
        "estimated_input_tokens": estimate_tokens(system) + estimate_tokens(user),
        "estimated_output_tokens": estimate_tokens(response),
        "paths": {
            "system": str(system_path),
            "user": str(user_path),
            "response": str(response_path),
        },
        "metadata": metadata or {},
    }
    budget_path.write_text(json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")
    return budget
PY

echo "[5/12] write reward_schema.py..."
cat > mvp/reward_schema.py <<'PY'
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import types
from typing import Any

import numpy as np


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

FORBIDDEN_SYNTAX = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.Try,
    ast.ClassDef,
    ast.Lambda,
    ast.Global,
    ast.Nonlocal,
)

REQUIRED_SIGNATURE = ["obs", "action", "next_obs", "done", "info"]


def build_default_schema(clean_interface: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "env_alias": clean_interface.get("env_alias"),
        "task_description": clean_interface.get("eureka_task_description", "")[:2000],
        "step_code": clean_interface.get("eureka_step_code", "")[:2000],
    }
    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

    return {
        "schema_version": f"eg_rsa_reward_schema_v1_{schema_hash}",
        "env_alias": clean_interface.get("env_alias", "Env-unknown"),
        "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
        "return_contract": "return float(total_reward), components_dict",
        "allowed_inputs": REQUIRED_SIGNATURE,
        "private_signal_policy": "Generated reward code must not use env_reward, fitness_score, or hidden evaluator details.",
        "components": [
            {
                "id": "progress",
                "description": "dense task-progress shaping inferred from public task context",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "stability",
                "description": "bounded shaping for stable/safe task-relevant behavior inferred from public task context",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "effort",
                "description": "bounded penalty for unnecessary or costly actions",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "terminal",
                "description": "bounded terminal shaping from public done signal",
                "direction": "maximize",
                "required": True,
            },
        ],
        "reward_abs_bound": 1000.0,
    }


def normalize_schema(raw: dict[str, Any] | None, clean_interface: dict[str, Any]) -> dict[str, Any]:
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


def _sample_obs(clean_interface: dict[str, Any]):
    if clean_interface.get("interface_mode") == "anonymous_clean":
        space = clean_interface.get("observation_space", {})
        if space.get("type") == "Box":
            shape = tuple(space.get("shape", [64]))
            return np.zeros(shape, dtype=np.float32)
    # Eureka step code commonly uses vector observations. Use a long generic vector for smoke test.
    return np.zeros((64,), dtype=np.float32)


def _sample_action(clean_interface: dict[str, Any]):
    return 0


def _find_compute_reward(tree: ast.AST) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
            return node
    return None


def _compile_reward(reward_code: str):
    tree = ast.parse(reward_code, mode="exec")
    module = types.ModuleType("reward_module")
    module.__dict__["np"] = np
    module.__dict__["math"] = math
    exec(compile(tree, filename="<reward_code>", mode="exec"), module.__dict__)
    return module.__dict__.get("compute_reward")


def validate_reward_code(
    reward_code: str,
    schema: dict[str, Any],
    clean_interface: dict[str, Any],
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    raw_lower = reward_code.lower()
    for name in FORBIDDEN_NAMES:
        if name.lower() in raw_lower:
            errors.append(f"forbidden token appears in code: {name}")

    try:
        tree = ast.parse(reward_code, mode="exec")
    except SyntaxError as e:
        return False, [f"syntax_error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_SYNTAX):
            errors.append(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            errors.append(f"forbidden attribute: {node.attr}")

    fn_node = _find_compute_reward(tree)
    if fn_node is None:
        errors.append("missing function: compute_reward")
    else:
        args = [a.arg for a in fn_node.args.args]
        if args != REQUIRED_SIGNATURE:
            errors.append(f"bad signature: expected {REQUIRED_SIGNATURE}, got {args}")

    if errors:
        return False, sorted(set(errors))

    try:
        fn = _compile_reward(reward_code)
        if fn is None:
            return False, ["compute_reward not found after compilation"]

        obs = _sample_obs(clean_interface)
        next_obs = copy.deepcopy(obs)
        action = _sample_action(clean_interface)

        out = fn(obs, action, next_obs, False, {})
        if not isinstance(out, tuple) or len(out) != 2:
            errors.append("compute_reward must return (total_reward, components_dict)")
        else:
            total, components = out
            total = float(total)
            if not np.isfinite(total):
                errors.append("total reward is not finite")
            if abs(total) > float(schema.get("reward_abs_bound", 1000.0)):
                errors.append("total reward exceeds reward_abs_bound")
            if not isinstance(components, dict):
                errors.append("components must be a dict")
            else:
                required_ids = [
                    c["id"]
                    for c in schema.get("components", [])
                    if c.get("required", False)
                ]
                missing = [cid for cid in required_ids if cid not in components]
                if missing:
                    errors.append(f"missing required components: {missing}")
                for k, v in components.items():
                    try:
                        fv = float(v)
                    except Exception:
                        errors.append(f"component {k} is not float-convertible")
                        continue
                    if not np.isfinite(fv):
                        errors.append(f"component {k} is not finite")

    except Exception as e:
        errors.append(f"smoke_test_error: {type(e).__name__}: {e}")

    return len(errors) == 0, sorted(set(errors))
PY

echo "[6/12] write lessons.py..."
cat > mvp/lessons.py <<'PY'
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_lesson(
    lesson: dict[str, Any],
    *,
    scope: str,
    env_alias: str,
    generation: int | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    out = dict(lesson)
    out.setdefault("lesson_id", f"{scope}_{uuid.uuid4().hex[:10]}")
    out.setdefault("scope", scope)
    out.setdefault("lesson_type", "general")
    out.setdefault("condition", "")
    out.setdefault("observation", "")
    out.setdefault("explanation", "")
    out.setdefault("recommendation", "")
    out.setdefault("confidence", 0.5)
    out.setdefault("reuse_policy", "same_env" if scope != "cross_environment" else "global")
    out.setdefault("env_alias", env_alias)
    if generation is not None:
        out.setdefault("generation", generation)
    if candidate_id is not None:
        out.setdefault("candidate_id", candidate_id)
    return out


def compact_lesson_line(row: dict[str, Any]) -> str:
    lid = row.get("lesson_id", "lesson")
    ltype = row.get("lesson_type", "general")
    cond = str(row.get("condition", "")).strip()
    rec = str(row.get("recommendation", "")).strip()
    conf = row.get("confidence", "")
    return f"- [{lid}] type={ltype}, confidence={conf}, condition={cond}, recommendation={rec}"


def retrieve_memory_context(
    *,
    stm_top: list[dict[str, Any]],
    candidate_lessons_path: Path,
    env_lessons_path: Path,
    ltm_lessons_path: Path,
    env_alias: str,
    candidate_lesson_top_k: int,
    env_lesson_top_k: int,
    ltm_lesson_top_k: int,
    max_chars: int,
) -> str:
    candidate_lessons = read_jsonl(candidate_lessons_path)
    env_lessons = read_jsonl(env_lessons_path)
    ltm_lessons = read_jsonl(ltm_lessons_path)

    parent_ids = {r.get("candidate_id") for r in stm_top}
    candidate_lessons = [
        x for x in candidate_lessons
        if x.get("candidate_id") in parent_ids or x.get("env_alias") == env_alias
    ][-candidate_lesson_top_k:]

    env_lessons = [
        x for x in env_lessons
        if x.get("env_alias") == env_alias
    ][-env_lesson_top_k:]

    ltm_lessons = [
        x for x in ltm_lessons
        if x.get("reuse_policy") in ("global", "similar_env", None)
    ][-ltm_lesson_top_k:]

    parts = []
    parts.append("Relevant candidate-level lessons:")
    parts.extend(compact_lesson_line(x) for x in candidate_lessons)
    if not candidate_lessons:
        parts.append("- none")

    parts.append("")
    parts.append("Relevant environment-level lessons:")
    parts.extend(compact_lesson_line(x) for x in env_lessons)
    if not env_lessons:
        parts.append("- none")

    parts.append("")
    parts.append("Relevant cross-environment lessons:")
    parts.extend(compact_lesson_line(x) for x in ltm_lessons)
    if not ltm_lessons:
        parts.append("- none")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def pack_generation_evidence(
    *,
    generation: int,
    records: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    gen_rows = [r for r in records if int(r.get("generation", -1)) == generation]
    ok_rows = [r for r in gen_rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: float(r.get("selection_score", -1e18)), reverse=True)

    def slim(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": r.get("candidate_id"),
            "parent_ids": r.get("parent_ids", []),
            "status": r.get("status"),
            "selection_score": r.get("selection_score"),
            "private_eval_return": r.get("hidden_eval_return"),
            "generated_return": r.get("train_mean_return"),
            "generated_minus_private": float(r.get("train_mean_return", 0.0)) - float(r.get("hidden_eval_return", 0.0)),
            "repair_attempts": r.get("repair_attempts", 0),
            "repair_success": r.get("repair_success", False),
            "validation_errors": r.get("validation_errors", []),
            "diagnostics": r.get("diagnostics", {}),
            "reward_code_head": str(r.get("reward_code", ""))[:2500],
            "llm_rationale": str(r.get("llm_rationale", ""))[:1000],
        }

    return {
        "generation": generation,
        "num_candidates": len(gen_rows),
        "num_ok": len(ok_rows),
        "top_candidates": [slim(r) for r in ok_rows[:top_k]],
        "bottom_candidates": [slim(r) for r in ok_rows[-top_k:]],
        "failed_candidates": [slim(r) for r in gen_rows if r.get("status") != "ok"],
    }
PY

echo "[7/12] write memory.py..."
cat > mvp/memory.py <<'PY'
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CandidateRecord:
    generation: int
    candidate_id: str
    parent_ids: list[str]

    schema_version: str
    env_alias: str
    status: str
    validation_errors: list[str]

    repair_attempts: int
    repair_success: bool
    validation_errors_before_repair: list[str]
    validation_errors_after_repair: list[str]

    reflection_summary: str
    reward_code: str
    llm_rationale: str

    train_mean_return: float
    hidden_eval_return: float
    selection_score: float

    judge_score: float
    judge_reason: str
    judge_details: dict[str, Any]
    video_path: str

    # Backward-compatible warning fields.
    identity_warning_count: int = 0
    identity_warnings: dict[str, Any] = field(default_factory=dict)
    semantic_term_warning_count: int = 0
    semantic_term_warnings: dict[str, Any] = field(default_factory=dict)
    semantic_warning_count: int = 0
    semantic_warnings: dict[str, Any] = field(default_factory=dict)

    # New EG-RSA fields.
    prompt_paths: dict[str, Any] = field(default_factory=dict)
    prompt_budgets: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    lesson_ids: list[str] = field(default_factory=list)


class JsonlMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CandidateRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def top_candidates(
        self,
        k: int,
        schema_version: Optional[str] = None,
        env_alias: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        rows = self.load_all()

        if schema_version is not None:
            rows = [r for r in rows if r.get("schema_version") == schema_version]
        if env_alias is not None:
            rows = [r for r in rows if r.get("env_alias") == env_alias]

        rows = [r for r in rows if r.get("status") == "ok"]
        rows.sort(key=lambda r: float(r.get("selection_score", -1e18)), reverse=True)
        return rows[:k]
PY

echo "[8/12] write agents.py..."
cat > mvp/agents.py <<'PY'
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_logging import write_llm_call
from .models import ModelGateway
from .reward_schema import build_default_schema, normalize_schema


PROMPT_DIR = Path(__file__).parent / "prompts"

PRIVATE_TERMS = (
    "env_reward",
    "hidden_env_reward",
    "_hidden_env_reward",
    "fitness_score",
    "compute_fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
)


@dataclass
class RewardDraft:
    candidate_id: str
    reward_code: str
    rationale: str
    llm_response: str
    prompt_budget: dict[str, Any]


@dataclass
class RepairDraft:
    reward_code: str
    rationale: str
    llm_response: str
    prompt_budget: dict[str, Any]


def _read_prompt(name: str, fallback: str) -> str:
    p = PROMPT_DIR / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return fallback


def _contains_private_term(text: str) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in PRIVATE_TERMS)


def _extract_code_and_rationale(text: str, stage: str) -> tuple[str, str]:
    if not text or not text.strip():
        raise ValueError(f"empty LLM response at stage={stage}")

    patterns = [
        r"```(?:python|py)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    code = ""
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            break

    if not code:
        m = re.search(r"(def\s+compute_reward\s*\(.*)", text, re.DOTALL)
        if m:
            code = m.group(1).strip()
        else:
            code = text.strip()

    if "def compute_reward" not in code:
        raise ValueError(
            f"could not extract compute_reward from LLM output at stage={stage}. "
            f"output_head={text[:500]!r}"
        )

    rationale_match = re.search(r"RATIONALE:(.*)", text, re.DOTALL | re.IGNORECASE)
    rationale = rationale_match.group(1).strip() if rationale_match else f"{stage} generated code"
    return code.strip(), rationale


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}

    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception:
        return {}


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass
    return []


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


class EnvUnderstandingAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "env_understanding_system.txt",
            "Read the Eureka task_description.txt and step.py. Produce an environment understanding report. "
            "You may reason about observation/action semantics. Do not reveal or use official reward formulas, "
            "env_reward, fitness_score, or hidden evaluator details.",
        )

    def analyze(self, clean_interface: dict[str, Any], log_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
        user = (
            "Eureka task_description.txt:\n"
            f"{clean_interface.get('eureka_task_description', '')}\n\n"
            "Eureka step.py:\n"
            f"{clean_interface.get('eureka_step_code', '')}\n\n"
            "Return a concise markdown report followed by a JSON object with keys: "
            "task_goal, observations, actions, termination, public_reward_design_variables, risks."
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "EnvUnderstandingAgent"})
        parsed = _extract_json_object(response)
        return response, parsed, budget


class SchemaPlannerAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "schema_planner_system.txt",
            "Create an environment-aware reward schema and search plan from the Eureka task context and environment understanding. "
            "The schema must define reward components for compute_reward(obs, action, next_obs, done, info). "
            "Do not use env_reward, fitness_score, official reward formulas, or hidden evaluator details.",
        )

    def plan(
        self,
        clean_interface: dict[str, Any],
        env_understanding: str,
        log_dir: Path,
    ) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        user = (
            "Eureka task_description.txt:\n"
            f"{clean_interface.get('eureka_task_description', '')}\n\n"
            "Eureka step.py:\n"
            f"{clean_interface.get('eureka_step_code', '')}\n\n"
            "Environment understanding:\n"
            f"{env_understanding}\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "reward_schema": {\n'
            '    "components": [{"id": "...", "description": "...", "direction": "maximize", "required": true}],\n'
            '    "reward_abs_bound": 1000.0\n'
            "  },\n"
            '  "search_plan": "markdown string"\n'
            "}\n"
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "SchemaPlannerAgent"})
        parsed = _extract_json_object(response)

        raw_schema = parsed.get("reward_schema", {}) if isinstance(parsed, dict) else {}
        schema = normalize_schema(raw_schema, clean_interface)

        search_plan = ""
        if isinstance(parsed, dict):
            search_plan = str(parsed.get("search_plan", "")).strip()
        if not search_plan:
            search_plan = response

        return schema, search_plan, response, budget


class RewardCoderAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "reward_coder_system.txt",
            "You are a reward engineer. Use Eureka task_description.txt and step.py plus EG-RSA memory/feedback "
            "to generate compute_reward(obs, action, next_obs, done, info). "
            "Output a Python function and a short RATIONALE. Do not use env_reward, fitness_score, official reward formulas, "
            "or hidden evaluator details.",
        )

    def draft(
        self,
        candidate_id: str,
        clean_interface: dict[str, Any],
        env_understanding: str,
        reward_schema: dict[str, Any],
        search_plan: str,
        feedback_context: str,
        memory_context: str,
        parent_codes: list[str],
        log_dir: Path,
        parent_code_max_chars: int,
    ) -> RewardDraft:
        safe_parent_codes = []
        for c in parent_codes:
            if _contains_private_term(c):
                continue
            safe_parent_codes.append(_truncate(c, parent_code_max_chars))

        parent_block = "\n\n".join(
            [f"Parent {i + 1}:\n```python\n{c}\n```" for i, c in enumerate(safe_parent_codes)]
        ) or "No parent reward code available."

        user = (
            f"Candidate ID: {candidate_id}\n\n"
            "Eureka task_description.txt:\n"
            f"{clean_interface.get('eureka_task_description', '')}\n\n"
            "Eureka step.py:\n"
            f"{clean_interface.get('eureka_step_code', '')}\n\n"
            "Environment understanding:\n"
            f"{env_understanding}\n\n"
            "Reward schema:\n"
            f"{json.dumps(reward_schema, ensure_ascii=False, indent=2)}\n\n"
            "Search plan:\n"
            f"{search_plan}\n\n"
            "Feedback context:\n"
            f"{feedback_context}\n\n"
            "Memory context:\n"
            f"{memory_context}\n\n"
            "Parent reward codes:\n"
            f"{parent_block}\n\n"
            "Generate one reward candidate. You may use task semantics from task_description.txt and step.py. "
            "The only hard output contract is compute_reward(obs, action, next_obs, done, info) returning float reward and components dict. "
            "Do not use env_reward, fitness_score, compute_fitness_score, official reward formulas, or hidden evaluator details."
        )

        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "RewardCoderAgent", "candidate_id": candidate_id})
        reward_code, rationale = _extract_code_and_rationale(response, stage="reward_generation")

        return RewardDraft(candidate_id, reward_code, rationale, response, budget)


class RepairAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "repair_system.txt",
            "Repair generated reward code so it satisfies the schema and validation contract. "
            "Do not use env_reward, fitness_score, compute_fitness_score, official reward formulas, or hidden evaluator details.",
        )

    def can_repair(self, reward_code: str, validation_errors: list[str]) -> bool:
        return not _contains_private_term(reward_code + "\n" + "\n".join(validation_errors))

    def repair(
        self,
        reward_code: str,
        validation_errors: list[str],
        reward_schema: dict[str, Any],
        attempt_index: int,
        log_dir: Path,
    ) -> RepairDraft:
        user = (
            f"Repair attempt: {attempt_index}\n\n"
            f"Validation errors:\n{validation_errors}\n\n"
            "Reward schema:\n"
            f"{json.dumps(reward_schema, ensure_ascii=False, indent=2)}\n\n"
            "Candidate reward code:\n"
            f"```python\n{reward_code}\n```\n\n"
            "Minimally repair schema, syntax, numerical stability, and return-contract issues only."
        )

        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "RepairAgent", "attempt": attempt_index})
        repaired_code, rationale = _extract_code_and_rationale(response, stage="repair")
        return RepairDraft(repaired_code, rationale, response, budget)


class ReflectionAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "reflection_system.txt",
            "Analyze structured reward-search evidence. Produce a concise reflection report for the next generation. "
            "Use private_eval_return only as a black-box selection score. Do not infer hidden evaluator formula.",
        )

    def reflect(
        self,
        evidence: dict[str, Any],
        previous_env_memory: str,
        memory_context: str,
        log_dir: Path,
    ) -> tuple[str, dict[str, Any]]:
        user = (
            "Structured evidence for this generation:\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            "Previous environment memory:\n"
            f"{previous_env_memory}\n\n"
            "Retrieved memory context:\n"
            f"{memory_context}\n\n"
            "Write:\n"
            "1. What worked.\n"
            "2. What failed.\n"
            "3. What to try next.\n"
            "4. Which lessons seem supported or contradicted.\n"
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "ReflectionAgent"})
        return response, budget


class LessonExtractorAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "lesson_extractor_system.txt",
            "Extract reusable lessons from structured evidence and reflection. Return JSON array only. "
            "Each lesson should include lesson_type, condition, observation, explanation, recommendation, confidence, reuse_policy.",
        )

    def extract(
        self,
        evidence: dict[str, Any],
        reflection_report: str,
        scope: str,
        env_alias: str,
        generation: int,
        log_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user = (
            f"Scope: {scope}\n"
            f"Env alias: {env_alias}\n"
            f"Generation: {generation}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            "Reflection report:\n"
            f"{reflection_report}\n\n"
            "Return JSON array of lessons. Do not include code blocks."
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "LessonExtractorAgent", "scope": scope})
        lessons = _extract_json_array(response)
        return lessons, budget


class VisionJudgeAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "vision_judge_system.txt",
            "Judge visible behavior quality only if visual evidence is available. Return JSON only.",
        )

    def judge(self, clean_interface: dict[str, Any], train_result: dict[str, Any], video_path) -> tuple[float, str, dict]:
        rubric = (
            f"Environment alias: {clean_interface.get('env_alias')}.\n"
            "Judge visible behavior quality only if visual evidence is available.\n"
            "Do not use private evaluator details or generated reward magnitude as proof of success.\n"
            "Return JSON only."
        )
        out = self.model.judge_video(self.system_prompt, rubric, video_path)
        score = float(out.get("score", 0.0))
        reason = str(out.get("reason", ""))
        if score <= 0:
            reason = reason or "no_visual_score_available"
        return max(0.0, min(100.0, score)), reason, out
PY

echo "[9/12] write rl_worker.py..."
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
from .env_sanitizer import ObservationAnonymizer


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
    def __init__(self, env: gym.Env, reward_fn: RewardFn, interface_mode: str = "eureka_clean"):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.interface_mode = interface_mode
        self._prev_obs_reward_view = None
        self._anonymizer = ObservationAnonymizer(env.observation_space)

    def _reward_view(self, obs):
        if self.interface_mode == "anonymous_clean":
            return self._anonymizer.transform(obs)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs_reward_view = self._reward_view(obs)
        return obs, info

    def step(self, action):
        next_obs, hidden_env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        next_obs_reward_view = self._reward_view(next_obs)

        out = self.reward_fn(
            self._prev_obs_reward_view,
            action,
            next_obs_reward_view,
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

        self._prev_obs_reward_view = next_obs_reward_view
        return next_obs, reward, terminated, truncated, safe_info


class RLWorker:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        self.interface_mode = getattr(cfg, "interface_mode", "eureka_clean")

    def train_and_eval(self, reward_code: str, ckpt_path: Path) -> dict[str, object]:
        reward_fn = compile_reward_function(reward_code)

        train_env = RewardFunctionWrapper(
            gym.make(self.cfg.env_id),
            reward_fn,
            interface_mode=self.interface_mode,
        )
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

        eval_env = RewardFunctionWrapper(
            gym.make(self.cfg.env_id),
            reward_fn,
            interface_mode=self.interface_mode,
        )

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

echo "[10/12] write orchestrator.py..."
cat > mvp/orchestrator.py <<'PY'
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .agents import (
    EnvUnderstandingAgent,
    LessonExtractorAgent,
    ReflectionAgent,
    RepairAgent,
    RewardCoderAgent,
    SchemaPlannerAgent,
    VisionJudgeAgent,
)
from .config import MVPConfig
from .env_sanitizer import infer_clean_env_interface
from .lessons import (
    append_jsonl,
    normalize_lesson,
    pack_generation_evidence,
    read_jsonl,
    retrieve_memory_context,
)
from .memory import CandidateRecord, JsonlMemory
from .models import ModelGateway
from .reward_schema import validate_reward_code
from .rl_worker import RLWorker
from .task_specs import get_private_task_spec, make_env_alias


MAX_REPAIR_ATTEMPTS = 2


class RewardEvolutionOrchestrator:
    def __init__(self, cfg: MVPConfig):
        self.cfg = cfg
        self.cfg.workspace.mkdir(parents=True, exist_ok=True)

        self.memory = JsonlMemory(cfg.memory_path)
        self.model = ModelGateway(cfg.model)

        self.env_understander = EnvUnderstandingAgent(self.model)
        self.schema_planner = SchemaPlannerAgent(self.model)
        self.coder = RewardCoderAgent(self.model)
        self.repairer = RepairAgent(self.model)
        self.reflector = ReflectionAgent(self.model)
        self.lesson_extractor = LessonExtractorAgent(self.model)
        self.judge = VisionJudgeAgent(self.model)
        self.worker = RLWorker(cfg.rl)

    def _write_json(self, path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> dict:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        private_task = get_private_task_spec(self.cfg.rl.env_id)
        env_alias = make_env_alias(private_task.env_id)
        interface_mode = getattr(self.cfg.rl, "interface_mode", "eureka_clean")

        clean_interface = infer_clean_env_interface(
            private_task.env_id,
            env_alias,
            interface_mode=interface_mode,
        )

        self._write_json(self.cfg.workspace / "clean_interface.txt", clean_interface)

        env_report, env_understanding_json, env_budget = self.env_understander.analyze(
            clean_interface,
            self.cfg.llm_dir / "bootstrap" / "env_understanding",
        )
        (self.cfg.artifacts_dir / "env_understanding.md").parent.mkdir(parents=True, exist_ok=True)
        (self.cfg.artifacts_dir / "env_understanding.md").write_text(env_report, encoding="utf-8")
        self._write_json(self.cfg.artifacts_dir / "env_understanding.json", env_understanding_json)

        reward_schema, search_plan, schema_raw_response, schema_budget = self.schema_planner.plan(
            clean_interface,
            env_report,
            self.cfg.llm_dir / "bootstrap" / "schema_planner",
        )
        self._write_json(self.cfg.workspace / "reward_schema.txt", reward_schema)
        (self.cfg.workspace / "clean_plan.txt").write_text(search_plan, encoding="utf-8")
        (self.cfg.artifacts_dir / "schema_planner_response.txt").write_text(schema_raw_response, encoding="utf-8")

        best: dict | None = None
        stagnant = 0
        best_score = float("-inf")
        feedback_context = "No prior generation feedback."

        for g in range(self.cfg.evolution.generations):
            top = self.memory.top_candidates(
                self.cfg.evolution.reflection_top_k,
                schema_version=reward_schema["schema_version"],
                env_alias=clean_interface["env_alias"],
            )
            parent_codes = [r["reward_code"] for r in top[: self.cfg.memory.parent_code_top_k]]
            parent_ids = [r["candidate_id"] for r in top]

            memory_context = retrieve_memory_context(
                stm_top=top,
                candidate_lessons_path=self.cfg.candidate_lessons_path,
                env_lessons_path=self.cfg.env_lessons_path,
                ltm_lessons_path=self.cfg.ltm_lessons_path,
                env_alias=clean_interface["env_alias"],
                candidate_lesson_top_k=self.cfg.memory.candidate_lesson_top_k,
                env_lesson_top_k=self.cfg.memory.env_lesson_top_k,
                ltm_lesson_top_k=self.cfg.memory.ltm_lesson_top_k,
                max_chars=self.cfg.memory.memory_context_max_chars,
            )
            gen_dir = self.cfg.artifacts_dir / f"generation_{g}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            (gen_dir / "memory_context.txt").write_text(memory_context, encoding="utf-8")
            (gen_dir / "feedback_context.txt").write_text(feedback_context, encoding="utf-8")

            generation_best = float("-inf")
            generation_records: list[dict] = []

            for i in range(self.cfg.evolution.population_size):
                cid = f"g{g}_c{i}"
                candidate_llm_dir = self.cfg.llm_dir / f"generation_{g}" / cid
                candidate_artifact_dir = self.cfg.artifacts_dir / f"generation_{g}" / cid
                candidate_artifact_dir.mkdir(parents=True, exist_ok=True)

                ckpt = self.cfg.checkpoints_dir / f"{cid}.zip"
                video = self.cfg.videos_dir / f"{cid}.gif"

                reward_code = ""
                rationale = ""
                validation_errors: list[str] = []
                validation_errors_before_repair: list[str] = []
                validation_errors_after_repair: list[str] = []
                repair_attempts = 0
                repair_success = False

                status = "failed"
                train_result = {
                    "eval_generated_return": -1e9,
                    "eval_hidden_return": -1e9,
                    "eval_episode_length": 0.0,
                    "component_returns": {},
                    "diagnostics": {},
                }
                judge_score = 0.0
                judge_reason = ""
                judge_details = {}
                prompt_budgets = {}
                prompt_paths = {}
                artifact_paths = {}

                try:
                    draft = self.coder.draft(
                        candidate_id=cid,
                        clean_interface=clean_interface,
                        env_understanding=env_report,
                        reward_schema=reward_schema,
                        search_plan=search_plan,
                        feedback_context=feedback_context[-self.cfg.memory.feedback_max_chars:],
                        memory_context=memory_context,
                        parent_codes=parent_codes,
                        log_dir=candidate_llm_dir / "reward_coder",
                        parent_code_max_chars=self.cfg.memory.parent_code_max_chars,
                    )
                    reward_code = draft.reward_code
                    rationale = draft.rationale
                    prompt_budgets["reward_coder"] = draft.prompt_budget
                    prompt_paths["reward_coder"] = draft.prompt_budget.get("paths", {})

                    (candidate_artifact_dir / "reward_code.py").write_text(reward_code, encoding="utf-8")
                    (candidate_artifact_dir / "rationale.txt").write_text(rationale, encoding="utf-8")
                    artifact_paths["reward_code"] = str(candidate_artifact_dir / "reward_code.py")

                    valid, validation_errors = validate_reward_code(
                        reward_code,
                        reward_schema,
                        clean_interface,
                    )

                    if not valid:
                        validation_errors_before_repair = list(validation_errors)

                        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
                            if not self.repairer.can_repair(reward_code, validation_errors):
                                break

                            repair_attempts = attempt
                            repair_draft = self.repairer.repair(
                                reward_code=reward_code,
                                validation_errors=validation_errors,
                                reward_schema=reward_schema,
                                attempt_index=attempt,
                                log_dir=candidate_llm_dir / f"repair_{attempt}",
                            )
                            reward_code = repair_draft.reward_code
                            rationale += f"\n\nREPAIR_ATTEMPT_{attempt}: {repair_draft.rationale}"
                            prompt_budgets[f"repair_{attempt}"] = repair_draft.prompt_budget
                            prompt_paths[f"repair_{attempt}"] = repair_draft.prompt_budget.get("paths", {})

                            (candidate_artifact_dir / f"repaired_reward_code_{attempt}.py").write_text(
                                reward_code,
                                encoding="utf-8",
                            )

                            valid, validation_errors = validate_reward_code(
                                reward_code,
                                reward_schema,
                                clean_interface,
                            )
                            if valid:
                                repair_success = True
                                break

                    validation_errors_after_repair = list(validation_errors)
                    self._write_json(
                        candidate_artifact_dir / "validation.json",
                        {
                            "valid": valid,
                            "validation_errors": validation_errors,
                            "validation_errors_before_repair": validation_errors_before_repair,
                            "validation_errors_after_repair": validation_errors_after_repair,
                        },
                    )

                    if not valid:
                        status = "invalid_schema"
                        judge_reason = "validation_error: " + "; ".join(validation_errors)
                    else:
                        train_result = self.worker.train_and_eval(reward_code, ckpt)
                        status = "ok"
                        self._write_json(candidate_artifact_dir / "train_result.json", train_result)

                        try:
                            self.worker.render_rollout_video(ckpt, video)
                            judge_score, judge_reason, judge_details = self.judge.judge(
                                clean_interface,
                                train_result,
                                video,
                            )
                        except Exception as e:
                            judge_score = 0.0
                            judge_reason = f"visual_judge_error: {type(e).__name__}: {e}"
                            judge_details = {"error": str(e)}

                except Exception as e:
                    status = "pipeline_error"
                    judge_reason = f"pipeline_error: {type(e).__name__}: {e}"
                    judge_details = {"error": str(e)}
                    rationale = rationale or "pipeline failed"

                private_eval_return = float(train_result.get("eval_hidden_return", -1e9))
                generated_return = float(train_result.get("eval_generated_return", -1e9))
                selection_score = private_eval_return if status == "ok" else -1e9

                rec = CandidateRecord(
                    generation=g,
                    candidate_id=cid,
                    parent_ids=parent_ids,
                    schema_version=reward_schema["schema_version"],
                    env_alias=clean_interface["env_alias"],
                    status=status,
                    validation_errors=validation_errors,
                    repair_attempts=repair_attempts,
                    repair_success=repair_success,
                    validation_errors_before_repair=validation_errors_before_repair,
                    validation_errors_after_repair=validation_errors_after_repair,
                    reflection_summary=feedback_context,
                    reward_code=reward_code,
                    llm_rationale=rationale,
                    train_mean_return=generated_return,
                    hidden_eval_return=private_eval_return,
                    selection_score=selection_score,
                    judge_score=float(judge_score),
                    judge_reason=judge_reason,
                    judge_details=judge_details,
                    video_path=str(video),
                    prompt_paths=prompt_paths,
                    prompt_budgets=prompt_budgets,
                    artifact_paths=artifact_paths,
                    diagnostics=dict(train_result.get("diagnostics", {})),
                    lesson_ids=[],
                )
                self.memory.append(rec)
                as_dict = rec.__dict__
                generation_records.append(as_dict)

                generation_best = max(generation_best, selection_score)
                if status == "ok" and (best is None or selection_score > best["selection_score"]):
                    best = as_dict

            all_records = self.memory.load_all()
            evidence = pack_generation_evidence(generation=g, records=all_records)
            self._write_json(gen_dir / "structured_evidence.json", evidence)

            previous_env_memory = self.cfg.env_memory_path.read_text(encoding="utf-8") if self.cfg.env_memory_path.exists() else ""

            try:
                reflection_report, reflection_budget = self.reflector.reflect(
                    evidence=evidence,
                    previous_env_memory=previous_env_memory,
                    memory_context=memory_context,
                    log_dir=self.cfg.llm_dir / f"generation_{g}" / "reflection",
                )
            except Exception as e:
                reflection_report = f"Reflection failed: {type(e).__name__}: {e}"
                reflection_budget = {}

            (gen_dir / "reflection_report.md").write_text(reflection_report, encoding="utf-8")
            feedback_context = reflection_report

            try:
                env_lessons_raw, env_lesson_budget = self.lesson_extractor.extract(
                    evidence=evidence,
                    reflection_report=reflection_report,
                    scope="environment",
                    env_alias=clean_interface["env_alias"],
                    generation=g,
                    log_dir=self.cfg.llm_dir / f"generation_{g}" / "lesson_extractor_env",
                )
            except Exception as e:
                env_lessons_raw = [
                    {
                        "lesson_type": "extractor_error",
                        "condition": "Lesson extraction failed.",
                        "observation": str(e),
                        "explanation": "The framework captured the error as a lesson.",
                        "recommendation": "Inspect lesson extractor prompt/response.",
                        "confidence": 0.2,
                        "reuse_policy": "same_env",
                    }
                ]

            env_lessons = [
                normalize_lesson(x, scope="environment", env_alias=clean_interface["env_alias"], generation=g)
                for x in env_lessons_raw
            ]
            append_jsonl(self.cfg.env_lessons_path, env_lessons)
            append_jsonl(self.cfg.ltm_lessons_path, [
                normalize_lesson(x, scope="cross_environment", env_alias=clean_interface["env_alias"], generation=g)
                for x in env_lessons_raw
                if str(x.get("reuse_policy", "")).lower() in ("global", "cross_environment", "similar_env")
            ])

            env_memory_text = (
                "# Environment Memory\n\n"
                f"env_alias: {clean_interface['env_alias']}\n"
                f"latest_generation: {g}\n\n"
                "## Latest reflection\n"
                f"{reflection_report}\n\n"
                "## Recent environment lessons\n"
                + "\n".join(
                    f"- {x.get('lesson_type')}: {x.get('recommendation')}"
                    for x in read_jsonl(self.cfg.env_lessons_path)[-20:]
                )
            )
            self.cfg.env_memory_path.write_text(env_memory_text, encoding="utf-8")

            if generation_best > best_score:
                best_score = generation_best
                stagnant = 0
            else:
                stagnant += 1

            if self.cfg.evolution.target_score is not None and best_score >= self.cfg.evolution.target_score:
                break

            if (
                self.cfg.evolution.max_stagnation_generations is not None
                and stagnant >= self.cfg.evolution.max_stagnation_generations
            ):
                break

        return best or {}


def format_report(best: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EG-RSA Reward Search Run Report",
        f"best_candidate: {best.get('candidate_id', 'N/A')}",
        f"schema_version: {best.get('schema_version', 'N/A')}",
        f"env_alias: {best.get('env_alias', 'N/A')}",
        f"status: {best.get('status', 'N/A')}",
        f"selection_score_private_eval: {best.get('selection_score', 0)}",
        f"private_eval_return: {best.get('hidden_eval_return', 0)}",
        f"generated_reward_return: {best.get('train_mean_return', 0)}",
        f"repair_attempts: {best.get('repair_attempts', 0)}",
        f"repair_success: {best.get('repair_success', False)}",
        f"judge_score: {best.get('judge_score', 0)}",
        f"judge_reason: {best.get('judge_reason', '')}",
        f"parents: {best.get('parent_ids', [])}",
        "",
        "## Reflection / Feedback Context",
        best.get("reflection_summary", ""),
        "",
        "## Diagnostics",
        "```json",
        json.dumps(best.get("diagnostics", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Prompt paths",
        "```json",
        json.dumps(best.get("prompt_paths", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Reward code",
        "```python",
        best.get("reward_code", ""),
        "```",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
PY

echo "[11/12] write exporters.py and patch models.py..."
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
                        "generated_private_gap": row.get("diagnostics", {}).get("generated_private_gap"),
                        "episode_length_mean": row.get("diagnostics", {}).get("episode_length_mean"),
                        "component_returns": row.get("diagnostics", {}).get("component_returns", {}),
                        "repair_attempts": row.get("repair_attempts", 0),
                        "repair_success": row.get("repair_success", False),
                        "judge_score": row.get("judge_score"),
                        "error_type": _error_type_from_reason(str(row.get("judge_reason", ""))),
                        "validation_errors": row.get("validation_errors", []),
                        "prompt_paths": row.get("prompt_paths", {}),
                        "artifact_paths": row.get("artifact_paths", {}),
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
        "generated_private_gap",
        "episode_length_mean",
        "component_returns",
        "repair_attempts",
        "repair_success",
        "judge_score",
        "error_type",
        "validation_errors",
        "prompt_paths",
        "artifact_paths",
        "judge_reason",
        "video_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv
PY

python - <<'PY'
from pathlib import Path

p = Path("mvp/models.py")
s = p.read_text(encoding="utf-8")

old = 'options={"temperature": self.config.temperature},'
new = '''options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    **({"num_ctx": self.config.ollama_num_ctx} if getattr(self.config, "ollama_num_ctx", None) else {}),
                },'''
if old in s:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
PY

echo "[12/12] write prompt files and syntax check..."
cat > mvp/prompts/env_understanding_system.txt <<'TXT'
You are an environment-understanding agent for reinforcement-learning reward design.

Input boundary:
- You are given Eureka-style task_description.txt and step.py.
- You may reason freely about task semantics, observation meanings, action meanings, termination conditions, and publicly visible state variables.
- Do not use env_reward, official reward formulas, fitness_score, compute_fitness_score, hidden evaluator implementation, or expert reward templates as generated reward logic.

Output:
- A concise markdown environment understanding report.
- Then a JSON object with keys:
  task_goal, observations, actions, termination, public_reward_design_variables, risks.
TXT

cat > mvp/prompts/schema_planner_system.txt <<'TXT'
You are a schema-planning agent for reinforcement-learning reward search.

Use the Eureka-style task description, step.py, and environment understanding to propose:
1. an environment-aware reward schema;
2. a search plan for reward generation.

You may use task semantics from task_description.txt and step.py.
Do not use env_reward, official reward formulas, fitness_score, compute_fitness_score, hidden evaluator implementation, or expert reward templates.

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
- Eureka step.py
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
- Do not use env_reward.
- Do not use fitness_score or compute_fitness_score.
- Do not use official reward formulas, hidden evaluator details, or expert reward templates.
- Do not import packages.
- Use only Python builtins, math, and numpy as np.

Output a Python code block containing compute_reward, followed by RATIONALE:<short explanation>.
TXT

cat > mvp/prompts/repair_system.txt <<'TXT'
You are a reward-code repair agent.

Minimally repair a generated reward function to satisfy:
- compute_reward(obs, action, next_obs, done, info)
- return float(total_reward), components_dict
- required schema components exist
- bounded finite numeric output
- no import statements
- no env_reward
- no fitness_score or compute_fitness_score
- no official reward formula or hidden evaluator details

Output a Python code block with the repaired function, followed by RATIONALE:<short explanation>.
TXT

cat > mvp/prompts/reflection_system.txt <<'TXT'
You are a reflection agent for iterative reward search.

Input:
- structured evidence from one generation
- previous environment memory
- retrieved STM/MTM/LTM lessons

Use private_eval_return only as a black-box selection score.
Do not infer the hidden evaluator formula.
Analyze what worked, what failed, what to try next, and which lessons are supported or contradicted.
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

Do not include code blocks.
TXT

python -m py_compile \
  mvp/config.py \
  mvp/env_sanitizer.py \
  mvp/llm_logging.py \
  mvp/reward_schema.py \
  mvp/lessons.py \
  mvp/memory.py \
  mvp/agents.py \
  mvp/rl_worker.py \
  mvp/orchestrator.py \
  mvp/exporters.py \
  mvp/models.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Do not run full multiseed yet."
echo "Next first check:"
echo "  python -m py_compile mvp/*.py"
echo ""
echo "Then run one tiny smoke:"
echo "  python run_mvp.py --config mvp/configs/lunar_lander_clean_deepseek_seed0.yaml --timesteps 1000"
