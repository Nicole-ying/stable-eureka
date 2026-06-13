#!/usr/bin/env bash
set -euo pipefail

echo "[1/9] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_clean_reward_patch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/agents.py \
  mvp/task_specs.py \
  mvp/rl_worker.py \
  mvp/orchestrator.py \
  mvp/models.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/prompts/reward_coder_system.txt \
  mvp/prompts/reflection_system.txt \
  mvp/prompts/vision_judge_system.txt
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/9] write clean task specs..."
cat > mvp/task_specs.py <<'PY'
from __future__ import annotations

import hashlib
from dataclasses import dataclass


# ============================================================
# PublicTaskSpec
# ============================================================
#
# 只允许进入 LLM prompt 的公开任务信息。
#
# 设计原则：
#   1. 不写真实 env_id 的语义解释；
#   2. 不写 observation 每一维的物理含义；
#   3. 不写 action 每个编号/维度的真实含义；
#   4. 不写官方 reward decomposition；
#   5. 不写 benchmark / fitness / hidden reward 相关内容。
#
# 这样做的目的：
#   避免 LLM 依靠已知 Gym 环境知识或官方奖励模板，
#   而是基于 clean interface + 训练反馈搜索 reward。
# ============================================================


@dataclass(frozen=True)
class PublicTaskSpec:
    task_goal: str
    task_style: str
    forbidden_terms: tuple[str, ...]


@dataclass(frozen=True)
class PrivateTaskSpec:
    env_id: str
    hidden_eval_source: str = "env_reward"
    benchmark_id: str | None = None


DEFAULT_FORBIDDEN_TERMS = (
    "env_reward",
    "fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "compute_fitness_score",
    "LunarLander",
    "BipedalWalker",
    "CartPole",
    "gymnasium.envs",
)


PUBLIC_TASK_SPECS: dict[str, PublicTaskSpec] = {
    "LunarLander-v3": PublicTaskSpec(
        task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
        task_style=(
            "Prefer dense, bounded shaping terms that encourage progress, stability, "
            "low unnecessary control effort, and appropriate terminal behavior."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
    "LunarLander-v2": PublicTaskSpec(
        task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
        task_style=(
            "Prefer dense, bounded shaping terms that encourage progress, stability, "
            "low unnecessary control effort, and appropriate terminal behavior."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
    "BipedalWalker-v3": PublicTaskSpec(
        task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
        task_style=(
            "Prefer dense, bounded shaping terms that encourage progress, stability, "
            "smooth transitions, low unnecessary control effort, and robust completion."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
    "CartPole-v1": PublicTaskSpec(
        task_goal="Control the agent to maintain task success for as long as possible.",
        task_style=(
            "Prefer bounded shaping terms that encourage stable observations, smooth transitions, "
            "and appropriate terminal penalties."
        ),
        forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
    ),
}


PRIVATE_TASK_SPECS: dict[str, PrivateTaskSpec] = {
    "LunarLander-v3": PrivateTaskSpec(env_id="LunarLander-v3", benchmark_id="LunarLander-v3"),
    "LunarLander-v2": PrivateTaskSpec(env_id="LunarLander-v2", benchmark_id="LunarLander-v2"),
    "BipedalWalker-v3": PrivateTaskSpec(env_id="BipedalWalker-v3", benchmark_id="BipedalWalker-v3"),
    "CartPole-v1": PrivateTaskSpec(env_id="CartPole-v1", benchmark_id="CartPole-v1"),
}


def make_env_alias(env_id: str) -> str:
    digest = hashlib.sha1(env_id.encode("utf-8")).hexdigest()[:8]
    return f"Env-{digest}"


def get_public_task_spec(env_id: str) -> PublicTaskSpec:
    return PUBLIC_TASK_SPECS.get(
        env_id,
        PublicTaskSpec(
            task_goal="Control the agent to complete the task successfully, stably, and efficiently.",
            task_style=(
                "Use clean observation/action interface information and closed-loop feedback. "
                "Do not assume any official reward template."
            ),
            forbidden_terms=DEFAULT_FORBIDDEN_TERMS,
        ),
    )


def get_private_task_spec(env_id: str) -> PrivateTaskSpec:
    return PRIVATE_TASK_SPECS.get(env_id, PrivateTaskSpec(env_id=env_id, benchmark_id=env_id))
PY

echo "[3/9] write clean env interface extractor..."
cat > mvp/env_sanitizer.py <<'PY'
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _clip_float_list(x, max_items: int = 32) -> list[float] | str:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size > max_items:
        return f"<omitted:{arr.size}_values>"
    out = []
    for v in arr.tolist():
        if np.isinf(v):
            out.append(float(v))
        else:
            out.append(round(float(v), 6))
    return out


def _space_to_public_dict(space: spaces.Space) -> dict[str, Any]:
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": _clip_float_list(space.low),
            "high": _clip_float_list(space.high),
        }

    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
        }

    if isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": np.asarray(space.nvec).astype(int).tolist(),
        }

    if isinstance(space, spaces.MultiBinary):
        return {
            "type": "MultiBinary",
            "n": int(space.n) if isinstance(space.n, int) else list(space.n),
        }

    return {
        "type": type(space).__name__,
        "repr": repr(space),
    }


def infer_clean_env_interface(env_id: str, env_alias: str) -> dict[str, Any]:
    """
    从真实 gym env 中抽取 clean interface。

    重要：
      1. 不返回真实 env_id；
      2. 不返回原始 env reward；
      3. 不返回源码、docstring、官方 reward 描述；
      4. 不返回 observation/action 的人工语义解释。
    """
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        interface = {
            "env_alias": env_alias,
            "observation_space": _space_to_public_dict(obs_space),
            "action_space": _space_to_public_dict(act_space),
            "api": {
                "reset": "obs, info = env.reset()",
                "step": "next_obs, hidden_env_reward, terminated, truncated, info = env.step(action)",
                "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
                "reward_return": "total_reward, components",
            },
            "visibility_policy": {
                "hidden_from_reward_designer": [
                    "real env_id",
                    "raw environment source code",
                    "docstrings and comments",
                    "original env_reward",
                    "official reward formula",
                    "fitness_score implementation",
                    "benchmark id",
                ],
                "visible_to_reward_designer": [
                    "anonymous env alias",
                    "observation space shape/range",
                    "action space type/shape",
                    "done flag",
                    "info dictionary keys only if observed at runtime",
                    "closed-loop training statistics",
                ],
            },
        }
        return interface
    finally:
        env.close()
PY

echo "[4/9] write reward schema and validator..."
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


def build_default_schema(public_task: dict[str, Any], clean_interface: dict[str, Any]) -> dict[str, Any]:
    """
    BootstrapAgent 生成初始 schema。

    这里故意用通用组件，不写任务专属物理语义：
      progress  : 泛化的任务进展代理；
      stability : 状态变化/姿态/数值稳定性代理；
      effort    : 动作幅值或动作切换成本；
      terminal  : done 时的终端 shaping。
    """
    payload = {
        "env_alias": clean_interface["env_alias"],
        "observation_space": clean_interface["observation_space"],
        "action_space": clean_interface["action_space"],
        "task_goal": public_task["task_goal"],
        "task_style": public_task["task_style"],
    }
    schema_hash = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

    return {
        "schema_version": f"clean_reward_schema_v1_{schema_hash}",
        "env_alias": clean_interface["env_alias"],
        "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
        "return_contract": "return float(total_reward), components_dict",
        "allowed_inputs": REQUIRED_SIGNATURE,
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "components": [
            {
                "id": "progress",
                "description": "dense task progress proxy inferred only from obs/action transitions",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "stability",
                "description": "bounded penalty for unstable or abrupt transitions",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "effort",
                "description": "bounded penalty for unnecessary action magnitude or switching",
                "direction": "maximize",
                "required": True,
            },
            {
                "id": "terminal",
                "description": "bounded terminal success/failure shaping without reading hidden reward",
                "direction": "maximize",
                "required": True,
            },
        ],
        "reward_abs_bound": 1000.0,
    }


def _sample_obs(space_dict: dict[str, Any]):
    if space_dict.get("type") == "Box":
        shape = tuple(space_dict.get("shape", []))
        return np.zeros(shape, dtype=np.float32)
    if space_dict.get("type") == "Discrete":
        return int(space_dict.get("start", 0))
    if space_dict.get("type") == "MultiDiscrete":
        return np.zeros(len(space_dict.get("nvec", [])), dtype=np.int64)
    if space_dict.get("type") == "MultiBinary":
        return np.zeros(space_dict.get("n", 1), dtype=np.int8)
    return np.zeros((1,), dtype=np.float32)


def _sample_action(space_dict: dict[str, Any]):
    if space_dict.get("type") == "Box":
        shape = tuple(space_dict.get("shape", []))
        return np.zeros(shape, dtype=np.float32)
    if space_dict.get("type") == "Discrete":
        return int(space_dict.get("start", 0))
    if space_dict.get("type") == "MultiDiscrete":
        return np.zeros(len(space_dict.get("nvec", [])), dtype=np.int64)
    if space_dict.get("type") == "MultiBinary":
        return np.zeros(space_dict.get("n", 1), dtype=np.int8)
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

        obs = _sample_obs(clean_interface["observation_space"])
        next_obs = copy.deepcopy(obs)
        action = _sample_action(clean_interface["action_space"])

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

echo "[5/9] write agents..."
cat > mvp/agents.py <<'PY'
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ModelGateway
from .reward_schema import build_default_schema
from .task_specs import PublicTaskSpec


PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass
class RewardDraft:
    candidate_id: str
    reward_code: str
    rationale: str


class BootstrapAgent:
    """
    Bootstrap schema agent.

    当前版本故意不读取原始 env.py，也不读取官方 reward。
    只基于 PublicTaskSpec + CleanEnvInterface 生成通用 RewardSchema。
    """

    def build_schema(
        self,
        public_task: PublicTaskSpec,
        clean_interface: dict[str, Any],
    ) -> dict[str, Any]:
        return build_default_schema(public_task.__dict__, clean_interface)


class PlannerAgent:
    """
    只生成干净 plan，不暴露真实 env_id，不解释 obs/action 物理含义。
    """

    def __init__(self):
        self.system_prompt = (PROMPT_DIR / "planner_system.txt").read_text(encoding="utf-8")

    def plan(
        self,
        public_task: PublicTaskSpec,
        clean_interface: dict[str, Any],
        reward_schema: dict[str, Any],
    ) -> str:
        return (
            f"Environment alias: {clean_interface['env_alias']}\n"
            f"Task goal: {public_task.task_goal}\n"
            f"Task style: {public_task.task_style}\n\n"
            "Clean interface:\n"
            f"- observation_space: {clean_interface['observation_space']}\n"
            f"- action_space: {clean_interface['action_space']}\n"
            f"- reward_signature: {reward_schema['reward_signature']}\n"
            f"- required_components: {[c['id'] for c in reward_schema['components'] if c.get('required')]}\n\n"
            "Important boundary:\n"
            "- Do not infer or mention the real environment name.\n"
            "- Do not use any original environment reward.\n"
            "- Do not use benchmark, official reward, or hidden fitness implementation.\n"
            "- Design reward only from obs/action/next_obs/done/info and closed-loop feedback."
        )


class RewardCoderAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = (PROMPT_DIR / "reward_coder_system.txt").read_text(encoding="utf-8")

    def draft(
        self,
        candidate_id: str,
        plan: str,
        clean_interface: dict[str, Any],
        reward_schema: dict[str, Any],
        reflection_context: str,
        parent_codes: list[str],
    ) -> RewardDraft:
        parent_block = "\n\n".join(
            [f"Parent {i + 1}:\n```python\n{c}\n```" for i, c in enumerate(parent_codes)]
        ) or "No clean parent code yet."

        schema_components = "\n".join(
            [
                f"- {c['id']}: {c['description']} | direction={c['direction']} | required={c['required']}"
                for c in reward_schema["components"]
            ]
        )

        user = (
            f"Candidate ID: {candidate_id}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Clean environment interface:\n{clean_interface}\n\n"
            f"Reward schema version: {reward_schema['schema_version']}\n"
            f"Reward signature: {reward_schema['reward_signature']}\n"
            f"Required schema components:\n{schema_components}\n\n"
            f"Forbidden names/tokens:\n{reward_schema['forbidden_names']}\n\n"
            f"Reflection from previous clean candidates:\n{reflection_context}\n\n"
            f"Parent reward codes from the same clean schema only:\n{parent_block}\n\n"
            "Now generate one reward function candidate that strictly follows the schema."
        )

        text = self.model.chat(self.system_prompt, user)
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        reward_code = code_match.group(1).strip() if code_match else text.strip()
        rationale_match = re.search(r"RATIONALE:(.*)", text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "LLM-generated clean reward candidate"
        return RewardDraft(candidate_id=candidate_id, reward_code=reward_code, rationale=rationale)


class VisionJudgeAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = (PROMPT_DIR / "vision_judge_system.txt").read_text(encoding="utf-8")

    def judge(
        self,
        clean_interface: dict[str, Any],
        train_result: dict[str, float],
        video_path,
    ) -> tuple[float, str, dict]:
        rubric = (
            f"Environment alias: {clean_interface['env_alias']}.\n"
            "Judge only visible behavior quality if visual evidence is available.\n"
            "Do not infer or mention the true environment name.\n"
            "Reward search selection is primarily based on hidden evaluator return, not generated reward return.\n"
            f"Hidden evaluator return: {train_result.get('eval_hidden_return', 0.0):.6f}\n"
            f"Generated reward return: {train_result.get('eval_generated_return', 0.0):.6f}\n"
        )

        out = self.model.judge_video(self.system_prompt, rubric, video_path)
        score = float(out.get("score", 0.0))
        reason = str(out.get("reason", ""))

        # 不再使用 generated reward return 兜底打分，避免 reward hacking。
        if score <= 0:
            reason = reason or "no_visual_score_available"

        return max(0.0, min(100.0, score)), reason, out


class ReflectionAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = (PROMPT_DIR / "reflection_system.txt").read_text(encoding="utf-8")

    def summarize(self, top_records: list[dict]) -> str:
        if not top_records:
            return (
                "No prior clean candidates. Start with bounded progress, stability, effort, "
                "and terminal components. Avoid environment-specific assumptions."
            )

        summary_lines = []
        for r in top_records:
            summary_lines.append(
                (
                    f"id={r.get('candidate_id')}, status={r.get('status')}, "
                    f"selection_score={r.get('selection_score')}, "
                    f"hidden_return={r.get('hidden_eval_return')}, "
                    f"generated_return={r.get('train_mean_return')}, "
                    f"reason={r.get('judge_reason')}, "
                    f"validation_errors={r.get('validation_errors', [])}"
                )
            )

        user = (
            "Past clean candidates from the same schema only:\n"
            + "\n".join(summary_lines)
            + "\n\nDo not infer the true environment name. Propose schema-preserving mutation hypotheses."
        )
        return self.model.chat(self.system_prompt, user)
PY

echo "[6/9] write RL worker without env_reward leakage..."
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
    """
    Compile LLM reward code into compute_reward(obs, action, next_obs, done, info).

    关键边界：
      - 不允许 env_reward 参数；
      - 不允许 import；
      - 不允许读取 hidden evaluator / benchmark / fitness；
      - 只能使用 np、math、python builtins。
    """
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

    fn = module.__dict__["compute_reward"]
    return fn


class RewardFunctionWrapper(gym.Wrapper):
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

        out = self.reward_fn(self._prev_obs, action, next_obs, done, info)
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

    def train_and_eval(self, reward_code: str, ckpt_path: Path) -> dict[str, float]:
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

        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            gen_ret = 0.0
            hid_ret = 0.0
            ep_len = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)
                gen_ret += float(reward)
                hid_ret += float(info.get("_hidden_env_reward", 0.0))
                ep_len += 1

            generated_returns.append(gen_ret)
            hidden_returns.append(hid_ret)
            episode_lengths.append(ep_len)

        eval_env.close()

        return {
            "eval_generated_return": float(np.mean(generated_returns)),
            "eval_hidden_return": float(np.mean(hidden_returns)),
            "eval_episode_length": float(np.mean(episode_lengths)),
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

echo "[7/9] write memory/exporters..."
cat > mvp/memory.py <<'PY'
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class CandidateRecord:
    generation: int
    candidate_id: str
    parent_ids: list[str]

    schema_version: str
    env_alias: str
    status: str
    validation_errors: list[str]

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
        schema_version: str | None = None,
        env_alias: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.load_all()

        # 关键：旧泄露 memory 没有 schema_version/env_alias/status 字段，自动被过滤。
        if schema_version is not None:
            rows = [r for r in rows if r.get("schema_version") == schema_version]
        if env_alias is not None:
            rows = [r for r in rows if r.get("env_alias") == env_alias]

        rows = [r for r in rows if r.get("status") == "ok"]
        rows.sort(key=lambda r: float(r.get("selection_score", -1e18)), reverse=True)
        return rows[:k]
PY

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
                        "hidden_eval_return": row.get("hidden_eval_return"),
                        "generated_return": row.get("train_mean_return"),
                        "judge_score": row.get("judge_score"),
                        "error_type": _error_type_from_reason(str(row.get("judge_reason", ""))),
                        "validation_errors": row.get("validation_errors", []),
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
        "hidden_eval_return",
        "generated_return",
        "judge_score",
        "error_type",
        "validation_errors",
        "judge_reason",
        "video_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv
PY

echo "[8/9] write orchestrator and model mock..."
cat > mvp/orchestrator.py <<'PY'
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .agents import BootstrapAgent, PlannerAgent, ReflectionAgent, RewardCoderAgent, VisionJudgeAgent
from .config import MVPConfig
from .env_sanitizer import infer_clean_env_interface
from .memory import CandidateRecord, JsonlMemory
from .models import ModelGateway
from .reward_schema import validate_reward_code
from .rl_worker import RLWorker
from .task_specs import get_private_task_spec, get_public_task_spec, make_env_alias


class RewardEvolutionOrchestrator:
    def __init__(self, cfg: MVPConfig):
        self.cfg = cfg
        self.cfg.workspace.mkdir(parents=True, exist_ok=True)

        self.memory = JsonlMemory(cfg.memory_path)
        self.model = ModelGateway(cfg.model)

        self.bootstrap = BootstrapAgent()
        self.planner = PlannerAgent()
        self.coder = RewardCoderAgent(self.model)
        self.judge = VisionJudgeAgent(self.model)
        self.reflector = ReflectionAgent(self.model)
        self.worker = RLWorker(cfg.rl)

    def run(self) -> dict:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        private_task = get_private_task_spec(self.cfg.rl.env_id)
        public_task = get_public_task_spec(private_task.env_id)
        env_alias = make_env_alias(private_task.env_id)

        clean_interface = infer_clean_env_interface(private_task.env_id, env_alias)
        reward_schema = self.bootstrap.build_schema(public_task, clean_interface)
        plan = self.planner.plan(public_task, clean_interface, reward_schema)

        self.cfg.workspace.mkdir(parents=True, exist_ok=True)
        (self.cfg.workspace / "clean_interface.txt").write_text(str(clean_interface), encoding="utf-8")
        (self.cfg.workspace / "reward_schema.txt").write_text(str(reward_schema), encoding="utf-8")
        (self.cfg.workspace / "clean_plan.txt").write_text(plan, encoding="utf-8")

        best: dict | None = None
        stagnant = 0
        best_score = float("-inf")

        for g in range(self.cfg.evolution.generations):
            top = self.memory.top_candidates(
                self.cfg.evolution.reflection_top_k,
                schema_version=reward_schema["schema_version"],
                env_alias=clean_interface["env_alias"],
            )
            parent_codes = [r["reward_code"] for r in top]
            parent_ids = [r["candidate_id"] for r in top]

            try:
                reflection = self.reflector.summarize(top)
            except Exception as e:
                reflection = f"reflection_error: {type(e).__name__}: {e}"

            generation_best = float("-inf")

            for i in range(self.cfg.evolution.population_size):
                cid = f"g{g}_c{i}"
                ckpt = self.cfg.checkpoints_dir / f"{cid}.zip"
                video = self.cfg.videos_dir / f"{cid}.gif"

                reward_code = ""
                rationale = ""
                validation_errors: list[str] = []
                status = "failed"
                train_result = {
                    "eval_generated_return": -1e9,
                    "eval_hidden_return": -1e9,
                    "eval_episode_length": 0.0,
                }
                judge_score = 0.0
                judge_reason = ""
                judge_details = {}

                try:
                    draft = self.coder.draft(
                        cid,
                        plan,
                        clean_interface,
                        reward_schema,
                        reflection,
                        parent_codes,
                    )
                    reward_code = draft.reward_code
                    rationale = draft.rationale

                    valid, validation_errors = validate_reward_code(
                        reward_code,
                        reward_schema,
                        clean_interface,
                    )

                    if not valid:
                        status = "invalid_schema"
                        judge_reason = "validation_error: " + "; ".join(validation_errors)
                    else:
                        train_result = self.worker.train_and_eval(reward_code, ckpt)
                        status = "ok"

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

                hidden_eval_return = float(train_result.get("eval_hidden_return", -1e9))
                generated_return = float(train_result.get("eval_generated_return", -1e9))
                selection_score = hidden_eval_return if status == "ok" else -1e9

                rec = CandidateRecord(
                    generation=g,
                    candidate_id=cid,
                    parent_ids=parent_ids,
                    schema_version=reward_schema["schema_version"],
                    env_alias=clean_interface["env_alias"],
                    status=status,
                    validation_errors=validation_errors,
                    reflection_summary=reflection,
                    reward_code=reward_code,
                    llm_rationale=rationale,
                    train_mean_return=generated_return,
                    hidden_eval_return=hidden_eval_return,
                    selection_score=selection_score,
                    judge_score=float(judge_score),
                    judge_reason=judge_reason,
                    judge_details=judge_details,
                    video_path=str(video),
                )
                self.memory.append(rec)
                as_dict = rec.__dict__

                generation_best = max(generation_best, selection_score)
                if status == "ok" and (best is None or selection_score > best["selection_score"]):
                    best = as_dict

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
        "# Clean Reward Search Run Report",
        f"best_candidate: {best.get('candidate_id', 'N/A')}",
        f"schema_version: {best.get('schema_version', 'N/A')}",
        f"env_alias: {best.get('env_alias', 'N/A')}",
        f"status: {best.get('status', 'N/A')}",
        f"selection_score_hidden_eval: {best.get('selection_score', 0)}",
        f"hidden_eval_return: {best.get('hidden_eval_return', 0)}",
        f"generated_reward_return: {best.get('train_mean_return', 0)}",
        f"judge_score: {best.get('judge_score', 0)}",
        f"judge_reason: {best.get('judge_reason', '')}",
        f"parents: {best.get('parent_ids', [])}",
        "",
        "## Reflection",
        best.get("reflection_summary", ""),
        "",
        "## Reward code",
        "```python",
        best.get("reward_code", ""),
        "```",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
PY

cat > mvp/models.py <<'PY'
import base64
import json
from pathlib import Path

import ollama
from openai import OpenAI

from .config import ModelConfig


class ModelGateway:
    """Thin model abstraction without external agent frameworks."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = config.provider.lower()
        self.openai_client = OpenAI() if self.provider == "openai" else None
        self.ollama_client = ollama.Client(host=config.ollama_host) if self.provider == "ollama" else None

    def chat(self, system: str, user: str) -> str:
        if self.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""

        if self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": self.config.temperature},
            )
            return response["message"]["content"]

        return (
            "```python\n"
            "def compute_reward(obs, action, next_obs, done, info):\n"
            "    obs_arr = np.asarray(obs, dtype=float).reshape(-1)\n"
            "    next_arr = np.asarray(next_obs, dtype=float).reshape(-1)\n"
            "    delta = next_arr - obs_arr\n"
            "    progress = float(np.clip(np.linalg.norm(obs_arr) - np.linalg.norm(next_arr), -5.0, 5.0))\n"
            "    stability = float(-0.05 * np.tanh(np.linalg.norm(delta)))\n"
            "    try:\n"
            "        act_arr = np.asarray(action, dtype=float).reshape(-1)\n"
            "        effort = float(-0.01 * np.tanh(np.linalg.norm(act_arr)))\n"
            "    except Exception:\n"
            "        effort = float(-0.01 * abs(float(action))) if isinstance(action, (int, float)) else 0.0\n"
            "    terminal = -1.0 if done else 0.0\n"
            "    total = progress + stability + effort + terminal\n"
            "    components = {\n"
            "        'progress': progress,\n"
            "        'stability': stability,\n"
            "        'effort': effort,\n"
            "        'terminal': terminal,\n"
            "    }\n"
            "    return float(total), components\n"
            "```\n"
            "RATIONALE: clean bounded transition reward without using hidden environment reward."
        )

    def judge_video(self, system_prompt: str, rubric: str, video_path: Path) -> dict:
        if self.provider == "openai":
            b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
            data_url = f"data:image/gif;base64,{b64}"
            response = self.openai_client.chat.completions.create(
                model=self.config.vlm_model,
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": rubric},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"score": 0.0, "reason": f"judge_parse_error: {content[:200]}"}

        if self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.config.vlm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{rubric}\nVideo file path: {video_path}. "
                            "If no visual access, return score 0 and explain limitation."
                        ),
                    },
                ],
                format="json",
                options={"temperature": 0.2},
            )
            try:
                return json.loads(response["message"]["content"])
            except json.JSONDecodeError:
                return {"score": 0.0, "reason": "ollama_json_parse_error"}

        return {"score": 0.0, "reason": "mock_judge_no_vision"}
PY

echo "[9/9] write clean prompts..."
mkdir -p mvp/prompts

cat > mvp/prompts/reward_coder_system.txt <<'TXT'
You are a reward-function search agent for reinforcement learning.

You must generate a clean reward function without access to any official reward, hidden evaluator, benchmark reward, or environment identity.

Output format:
```python
def compute_reward(obs, action, next_obs, done, info):
    ...
    return float(total_reward), components
```
RATIONALE:<one short paragraph>

Hard constraints:
- Must define exactly: compute_reward(obs, action, next_obs, done, info)
- Do not add env_reward to the signature.
- Do not use any variable or key named env_reward, hidden_reward, original_reward, official_reward, benchmark_reward, fitness_score, or compute_fitness_score.
- Use only Python builtins, math, and numpy symbols already available as np.
- No import statements.
- Return a tuple: (scalar float reward, dict of scalar float components).
- The components dict must contain all required schema component IDs.
- Keep reward magnitude bounded.
- Do not infer or mention the real environment name.
- Design from clean observation/action interface, transitions, done flag, info, parent candidates, and closed-loop feedback only.
TXT

cat > mvp/prompts/reflection_system.txt <<'TXT'
You are a reflection agent for clean autonomous reward evolution.

Summarize why top clean candidates worked or failed, using only:
- schema-aligned component behavior,
- validation status,
- generated reward return as diagnostic only,
- hidden evaluator return as the selection metric,
- judge comments if available.

Never infer or mention the real environment name.
Never propose using official reward, original env_reward, benchmark reward, or hidden fitness implementation.

Return:
1) What to keep
2) What to change
3) Next schema-preserving mutation hypotheses, max 5
TXT

cat > mvp/prompts/vision_judge_system.txt <<'TXT'
You are a strict visual RL policy judge.

Given anonymous task/interface evidence and rollout evidence, score behavior quality if visual evidence is actually available.

Do not infer or mention the true environment name.
Do not use generated reward return as proof of task success.

Return JSON keys:
- score: float from 0 to 100
- reason: short string
- strengths: array of short strings
- weaknesses: array of short strings
TXT

python -m py_compile \
  mvp/task_specs.py \
  mvp/env_sanitizer.py \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/rl_worker.py \
  mvp/orchestrator.py \
  mvp/models.py \
  mvp/memory.py \
  mvp/exporters.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Recommended smoke test:"
echo "  python run_mvp.py --config mvp/configs/cartpole_mock.yaml --provider mock --timesteps 2000"
echo ""
echo "Recommended clean LunarLander run:"
echo "  python run_mvp.py --config mvp/configs/lunar_lander.yaml --provider ollama --timesteps 30000"
