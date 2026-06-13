#!/usr/bin/env bash
set -euo pipefail

echo "[1/9] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_general_semantic_anonymization_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/env_sanitizer.py \
  mvp/rl_worker.py \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/orchestrator.py \
  mvp/prompts/reward_coder_system.txt \
  mvp/prompts/repair_system.txt \
  mvp/prompts/reflection_system.txt \
  scripts/audit_clean_run.py \
  scripts/summarize_clean_multiseed.py
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/9] write general semantic audit module..."
cat > mvp/semantic_audit.py <<'PY'
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# Generic semantic warning terms
# ============================================================
#
# 注意：
#   1. 这些词不是 hard leak terms，不直接导致实验失败；
#   2. 它们用于统计 LLM 是否在匿名接口下仍然给 obs 维度赋予物理语义；
#   3. 不只针对 CartPole/LunarLander，而是覆盖通用控制任务中常见的
#      物理/benchmark/目标语义词。
# ============================================================


SEMANTIC_WARNING_TERMS = (
    # benchmark / object identity style terms
    "cart",
    "pole",
    "lander",
    "landing",
    "leg",
    "contact",
    "mountain",
    "car",
    "pendulum",
    "acrobot",
    "walker",
    "bipedal",
    "hopper",
    "cheetah",
    "ant",
    "humanoid",
    "robot",

    # physical coordinate semantics
    "position",
    "velocity",
    "angle",
    "angular",
    "coordinate",
    "x coordinate",
    "y coordinate",
    "height",
    "altitude",
    "distance",
    "origin",
    "target",
    "goal",

    # dynamics / task outcome semantics
    "upright",
    "balance",
    "balancing",
    "fall",
    "falling",
    "crash",
    "land",
    "landed",
    "success flag",
    "failure",
    "thruster",
    "engine",
    "torque",
    "speed",
    "acceleration",
)


def _count_terms(text: str) -> dict[str, int]:
    lower = text.lower()
    counts: dict[str, int] = {}

    for term in SEMANTIC_WARNING_TERMS:
        # word-ish boundary for normal words, substring fallback for multi-word phrases
        if " " in term:
            n = lower.count(term.lower())
        else:
            n = len(re.findall(rf"\b{re.escape(term.lower())}\b", lower))
        if n:
            counts[term] = n

    return counts


def audit_semantic_text_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    per_artifact = {}
    total = 0

    for name, value in bundle.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        counts = _count_terms(text)
        per_artifact[name] = counts
        total += sum(counts.values())

    return {
        "semantic_warning_count": total,
        "semantic_warnings": per_artifact,
    }


def save_semantic_audit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[3/9] write general anonymous env_sanitizer.py..."
cat > mvp/env_sanitizer.py <<'PY'
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _finite_ratio(x) -> float:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def _range_category_box(space: spaces.Box) -> str:
    low_finite = _finite_ratio(space.low)
    high_finite = _finite_ratio(space.high)

    if low_finite == 1.0 and high_finite == 1.0:
        return "fully_bounded"
    if low_finite == 0.0 and high_finite == 0.0:
        return "unbounded"
    return "partially_bounded"


def _space_to_public_dict(space: spaces.Space) -> dict[str, Any]:
    """
    Return an anonymized public space description.

    关键原则：
      - 不暴露具体 low/high 数值，避免 LLM 通过 benchmark bounds 反推环境；
      - 不暴露 observation 维度语义；
      - 只暴露实现 reward code 必需的 shape/type/cardinality。
    """
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "range_category": _range_category_box(space),
            "value_view": "normalized_anonymous_values",
            "normalization": "finite Box dimensions are scaled approximately into [-1, 1]; non-finite dimensions are clipped by a generic scale.",
            "dimension_semantics": "not_available",
        }

    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
            "dimension_semantics": "not_available",
        }

    if isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "shape": list(np.asarray(space.nvec).shape),
            "cardinality_view": "available_without_semantic_labels",
            "dimension_semantics": "not_available",
        }

    if isinstance(space, spaces.MultiBinary):
        return {
            "type": "MultiBinary",
            "shape": list(np.asarray(space.n).shape) if not isinstance(space.n, int) else [int(space.n)],
            "dimension_semantics": "not_available",
        }

    return {
        "type": type(space).__name__,
        "description": "custom_space_anonymized",
        "dimension_semantics": "not_available",
    }


def infer_clean_env_interface(env_id: str, env_alias: str) -> dict[str, Any]:
    """
    抽取只允许进入 LLM 的 clean interface。

    通用匿名化原则：
      - 真实环境标识只在 private runtime 中使用；
      - 原始环境返回的私有评估信号不出现在这里；
      - 源码、docstring、官方奖励说明、fitness 实现都不出现在这里；
      - observation/action 不提供物理语义；
      - observation 不提供具体 low/high benchmark bounds；
      - reward function 看到的是 normalized anonymous observations。
    """
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space

        return {
            "env_alias": env_alias,
            "observation_space": _space_to_public_dict(obs_space),
            "action_space": _space_to_public_dict(act_space),
            "reward_observation_view": {
                "obs": "normalized anonymous observation with the same shape as the original observation",
                "next_obs": "normalized anonymous next observation with the same shape as the original observation",
                "raw_observation_semantics": "not_available",
            },
            "api": {
                "reset": "obs, info = reset()",
                "step_visible_outputs": [
                    "next_obs",
                    "terminated",
                    "truncated",
                    "info",
                ],
                "reward_signature": "compute_reward(obs, action, next_obs, done, info)",
                "reward_return": "total_reward, components",
            },
        }
    finally:
        env.close()
PY

echo "[4/9] patch reward_schema wording..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/reward_schema.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    '"description": "dense task progress proxy inferred only from obs/action transitions"',
    '"description": "dense progress proxy over anonymous normalized transition features"',
)
s = s.replace(
    '"description": "bounded penalty for unstable or abrupt transitions"',
    '"description": "bounded penalty for abrupt changes in anonymous normalized transition features"',
)
s = s.replace(
    '"description": "bounded penalty for unnecessary action magnitude or switching"',
    '"description": "bounded penalty for unnecessary action magnitude or action changes when observable"',
)
s = s.replace(
    '"description": "bounded terminal success/failure shaping from public termination signals"',
    '"description": "bounded terminal shaping from public termination signals only"',
)

p.write_text(s, encoding="utf-8")
PY

echo "[5/9] write rl_worker.py with generic normalized reward observations..."
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
from gymnasium import spaces
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


class ObservationAnonymizer:
    """
    Generic observation anonymizer for reward computation.

    策略：
      - policy 仍然看 raw obs；
      - reward function 只看 normalized anonymous obs；
      - 对 Box 空间：有限 bounds 用中心/半径归一化到约 [-1,1]；
        非有限维度用通用尺度 tanh 压缩；
      - 对非 Box 空间：转成 float array，做通用尺度压缩；
      - 不包含任何环境专属逻辑。
    """

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

            # 非有限 bound 维度不暴露原始尺度，用 generic tanh 压缩。
            nonfinite = ~finite
            out[nonfinite] = np.tanh(arr[nonfinite] / 5.0)

            out = np.clip(out, -5.0, 5.0)
            return out.astype(np.float32)

        return np.tanh(arr / 5.0).astype(np.float32)


class RewardFunctionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs = None
        self._prev_obs_reward_view = None
        self._anonymizer = ObservationAnonymizer(env.observation_space)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        self._prev_obs_reward_view = self._anonymizer.transform(obs)
        return obs, info

    def step(self, action):
        next_obs, hidden_env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        next_obs_reward_view = self._anonymizer.transform(next_obs)

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

        self._prev_obs = next_obs
        self._prev_obs_reward_view = next_obs_reward_view
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

echo "[6/9] harden prompts against observation semantic naming..."
cat > mvp/prompts/reward_coder_system.txt <<'TXT'
You are a reward-function search agent for reinforcement learning.

Generate a reward function using only the public transition interface and the provided schema. Do not infer the real environment identity. Do not rely on private runtime signals, benchmark signals, implementation internals, or private scoring logic.

The observation values are anonymous normalized features. Do not assign physical or benchmark-specific meanings to observation dimensions. Refer to them only as obs[i] and next_obs[i], or as anonymous feature indices. Do not describe dimensions as positions, velocities, angles, contacts, legs, targets, landing states, poles, carts, bodies, coordinates, or other task-specific entities.

Output format:
```python
def compute_reward(obs, action, next_obs, done, info):
    ...
    return float(total_reward), components
```
RATIONALE:<one short paragraph>

Hard constraints:
- Must define exactly: compute_reward(obs, action, next_obs, done, info)
- Use only Python builtins, math, and numpy symbols already available as np.
- No import statements.
- Return a tuple: (scalar float reward, dict of scalar float components).
- The components dict must contain all required schema component IDs.
- Keep reward magnitude bounded.
- Do not infer or mention the real environment identity.
- Do not assign semantic names to observation dimensions.
- Design from anonymous normalized observations, actions, transitions, done flag, info, parent candidates, and closed-loop feedback only.
TXT

cat > mvp/prompts/repair_system.txt <<'TXT'
You are a repair agent for generated reinforcement-learning reward code.

Your job is not to design a new reward from scratch. Your job is to minimally repair a candidate so that it satisfies the provided schema and validation contract.

The observation values are anonymous normalized features. Do not assign physical or benchmark-specific meanings to observation dimensions. Refer to them only as obs[i] and next_obs[i], or as anonymous feature indices.

Output format:
```python
def compute_reward(obs, action, next_obs, done, info):
    ...
    return float(total_reward), components
```
RATIONALE:<one short paragraph>

Hard constraints:
- Preserve the required function signature exactly.
- Return exactly a tuple: (scalar float reward, dict of scalar float components).
- Include all required component IDs in the components dict.
- Use only anonymous normalized observations, actions, transitions, done flag, and info.
- Use only Python builtins, math, and numpy symbols already available as np.
- No import statements.
- Avoid try/except, class definitions, global state, file IO, network IO, random seeds, or environment construction.
- Keep reward magnitude bounded and finite.
- Do not infer or mention the real environment identity.
- Do not assign semantic names to observation dimensions.
- Do not introduce private runtime signals, benchmark signals, implementation internals, or private scoring logic.
TXT

cat > mvp/prompts/reflection_system.txt <<'TXT'
You are a reflection agent for clean autonomous reward evolution.

Summarize why top clean candidates worked or failed, using only:
- schema-aligned component behavior,
- validation status,
- generated reward return as diagnostic only,
- private evaluation return as the selection metric,
- judge comments if available.

Observation values are anonymous normalized features. Do not infer observation dimension semantics. Do not describe dimensions as physical coordinates, velocities, angles, contacts, bodies, targets, or benchmark-specific entities.

Never infer or mention the real environment identity.
Never propose using private runtime signals, benchmark signals, implementation internals, or private scoring logic.

Return:
1) What to keep
2) What to change
3) Next schema-preserving mutation hypotheses, max 5
TXT

echo "[7/9] patch memory/exporter/orchestrator for semantic warnings..."
python - <<'PY'
from pathlib import Path

# memory.py
p = Path("mvp/memory.py")
s = p.read_text(encoding="utf-8")
if "semantic_warning_count" not in s:
    s = s.replace(
        "    validation_errors_after_repair: list[str]\n\n"
        "    reflection_summary: str\n",
        "    validation_errors_after_repair: list[str]\n\n"
        "    semantic_warning_count: int\n"
        "    semantic_warnings: dict[str, Any]\n\n"
        "    reflection_summary: str\n",
    )
p.write_text(s, encoding="utf-8")

# exporters.py
p = Path("mvp/exporters.py")
s = p.read_text(encoding="utf-8")
if '"semantic_warning_count": row.get("semantic_warning_count", 0),' not in s:
    s = s.replace(
        '                        "validation_errors_after_repair": row.get("validation_errors_after_repair", []),\n'
        '                        "judge_reason": row.get("judge_reason", ""),\n',
        '                        "validation_errors_after_repair": row.get("validation_errors_after_repair", []),\n'
        '                        "semantic_warning_count": row.get("semantic_warning_count", 0),\n'
        '                        "semantic_warnings": row.get("semantic_warnings", {}),\n'
        '                        "judge_reason": row.get("judge_reason", ""),\n',
    )
    s = s.replace(
        '        "validation_errors_after_repair",\n'
        '        "judge_reason",\n',
        '        "validation_errors_after_repair",\n'
        '        "semantic_warning_count",\n'
        '        "semantic_warnings",\n'
        '        "judge_reason",\n',
    )
p.write_text(s, encoding="utf-8")

# orchestrator.py
p = Path("mvp/orchestrator.py")
s = p.read_text(encoding="utf-8")

if "from .semantic_audit import audit_semantic_text_bundle" not in s:
    s = s.replace(
        "from .reward_schema import validate_reward_code\n",
        "from .reward_schema import validate_reward_code\n"
        "from .semantic_audit import audit_semantic_text_bundle, save_semantic_audit_report\n",
    )

if "semantic_pre_report = audit_semantic_text_bundle" not in s:
    s = s.replace(
        "        save_audit_report(audit, self.cfg.workspace / \"leak_audit_pre_generation.json\")\n"
        "        if not audit[\"ok\"]:\n",
        "        save_audit_report(audit, self.cfg.workspace / \"leak_audit_pre_generation.json\")\n"
        "        semantic_pre_report = audit_semantic_text_bundle(\n"
        "            {\n"
        "                \"clean_interface\": clean_interface,\n"
        "                \"reward_schema\": reward_schema,\n"
        "                \"clean_plan\": plan,\n"
        "            }\n"
        "        )\n"
        "        save_semantic_audit_report(\n"
        "            semantic_pre_report,\n"
        "            self.cfg.workspace / \"semantic_audit_pre_generation.json\",\n"
        "        )\n"
        "        if not audit[\"ok\"]:\n",
    )

if "semantic_report = audit_semantic_text_bundle" not in s:
    s = s.replace(
        "                private_eval_return = float(train_result.get(\"eval_hidden_return\", -1e9))\n",
        "                semantic_report = audit_semantic_text_bundle(\n"
        "                    {\n"
        "                        \"reward_code\": reward_code,\n"
        "                        \"llm_rationale\": rationale,\n"
        "                        \"reflection_summary\": reflection,\n"
        "                    }\n"
        "                )\n"
        "\n"
        "                private_eval_return = float(train_result.get(\"eval_hidden_return\", -1e9))\n",
    )

if "semantic_warning_count=semantic_report" not in s:
    s = s.replace(
        "                    validation_errors_after_repair=validation_errors_after_repair,\n"
        "                    reflection_summary=reflection,\n",
        "                    validation_errors_after_repair=validation_errors_after_repair,\n"
        "                    semantic_warning_count=int(semantic_report.get(\"semantic_warning_count\", 0)),\n"
        "                    semantic_warnings=semantic_report.get(\"semantic_warnings\", {}),\n"
        "                    reflection_summary=reflection,\n",
    )

if "semantic_warning_count:" not in s:
    s = s.replace(
        '        f"repair_success: {best.get(\'repair_success\', False)}",\n'
        '        f"judge_score: {best.get(\'judge_score\', 0)}",\n',
        '        f"repair_success: {best.get(\'repair_success\', False)}",\n'
        '        f"semantic_warning_count: {best.get(\'semantic_warning_count\', 0)}",\n'
        '        f"judge_score: {best.get(\'judge_score\', 0)}",\n',
    )

p.write_text(s, encoding="utf-8")
PY

echo "[8/9] patch audit and summary scripts for semantic warnings..."
python - <<'PY'
from pathlib import Path

# scripts/audit_clean_run.py
p = Path("scripts/audit_clean_run.py")
if p.exists():
    s = p.read_text(encoding="utf-8")
    if "semantic_warning_count_total" not in s:
        s = s.replace(
            '    repair_success = 0\n',
            '    repair_success = 0\n'
            '    semantic_warning_count_total = 0\n',
        )
        s = s.replace(
            '        repair_success += int(bool(row.get("repair_success", False)))\n',
            '        repair_success += int(bool(row.get("repair_success", False)))\n'
            '        semantic_warning_count_total += int(row.get("semantic_warning_count", 0) or 0)\n',
        )
        s = s.replace(
            '        "repair_success_count": repair_success,\n',
            '        "repair_success_count": repair_success,\n'
            '        "semantic_warning_count_total": semantic_warning_count_total,\n',
        )
    p.write_text(s, encoding="utf-8")

# scripts/summarize_clean_multiseed.py
p = Path("scripts/summarize_clean_multiseed.py")
if p.exists():
    s = p.read_text(encoding="utf-8")
    if '"semantic_warning_count_total":' not in s:
        s = s.replace(
            '        "repair_success_count": sum(int(bool(r.get("repair_success", False))) for r in rows),\n',
            '        "repair_success_count": sum(int(bool(r.get("repair_success", False))) for r in rows),\n'
            '        "semantic_warning_count_total": sum(int(r.get("semantic_warning_count", 0) or 0) for r in rows),\n',
        )
        s = s.replace(
            '                "mean_repair_success": mean([int(r["repair_success_count"]) for r in group]) if group else "",\n',
            '                "mean_repair_success": mean([int(r["repair_success_count"]) for r in group]) if group else "",\n'
            '                "mean_semantic_warning_count": mean([int(r.get("semantic_warning_count_total", 0)) for r in group]) if group else "",\n',
        )
    p.write_text(s, encoding="utf-8")
PY

echo "[9/9] syntax check..."
python -m py_compile \
  mvp/semantic_audit.py \
  mvp/env_sanitizer.py \
  mvp/rl_worker.py \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/orchestrator.py

if [ -f scripts/audit_clean_run.py ]; then
  python -m py_compile scripts/audit_clean_run.py
fi
if [ -f scripts/summarize_clean_multiseed.py ]; then
  python -m py_compile scripts/summarize_clean_multiseed.py
fi

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Recommended next:"
echo "  bash scripts/run_clean_cartpole_deepseek_multiseed.sh"
echo "  bash scripts/run_clean_lunar_deepseek_multiseed.sh"
echo ""
echo "Then inspect semantic warnings:"
echo "  cat runs/clean_cartpole_deepseek_seed0_g2p2_t8k/semantic_audit_pre_generation.json"
echo "  cat runs/clean_cartpole_deepseek_seed0_g2p2_t8k/memory.csv"
