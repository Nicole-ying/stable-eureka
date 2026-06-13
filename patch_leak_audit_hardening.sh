#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_leak_audit_patch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/env_sanitizer.py \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/orchestrator.py \
  mvp/prompts/reward_coder_system.txt \
  mvp/prompts/reflection_system.txt \
  mvp/prompts/vision_judge_system.txt
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/7] add leak audit module..."
cat > mvp/leak_audit.py <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class LeakAuditError(RuntimeError):
    pass


# ============================================================
# Leak audit terms
# ============================================================
#
# 这些词只用于本地审计，不进入 RewardCoder prompt。
#
# 目标：
#   1. clean_interface / clean_plan / public_schema 不能出现真实环境名；
#   2. 不能出现原始奖励、隐藏评估、benchmark、fitness 等私有信号的显式变量名；
#   3. 不能出现旧 Gym 任务名，避免 LLM 用通用知识套官方 reward。
# ============================================================


DEFAULT_LEAK_TERMS = (
    "env_reward",
    "hidden_env_reward",
    "_hidden_env_reward",
    "fitness_score",
    "compute_fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
    "LunarLander",
    "BipedalWalker",
    "CartPole",
    "Acrobot",
    "MountainCar",
    "Pendulum",
)


def _normalize_terms(
    env_id: str | None = None,
    extra_terms: Iterable[str] | None = None,
) -> list[str]:
    terms = set(DEFAULT_LEAK_TERMS)

    if env_id:
        terms.add(env_id)
        terms.add(env_id.lower())
        terms.add(env_id.replace("-", ""))
        terms.add(env_id.replace("-", "_"))
        terms.add(env_id.split("-")[0])

    if extra_terms:
        for term in extra_terms:
            if term:
                terms.add(str(term))

    return sorted(t for t in terms if t)


def audit_text_bundle(
    bundle: dict[str, Any],
    env_id: str | None = None,
    extra_terms: Iterable[str] | None = None,
) -> dict[str, Any]:
    terms = _normalize_terms(env_id=env_id, extra_terms=extra_terms)
    violations = []

    for name, value in bundle.items():
        text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
        text_lower = text.lower()

        for term in terms:
            if term.lower() in text_lower:
                violations.append(
                    {
                        "artifact": name,
                        "term": term,
                    }
                )

    return {
        "ok": len(violations) == 0,
        "num_violations": len(violations),
        "violations": violations,
    }


def assert_no_leak_text(
    name: str,
    text: str,
    env_id: str | None = None,
    extra_terms: Iterable[str] | None = None,
) -> None:
    audit = audit_text_bundle({name: text}, env_id=env_id, extra_terms=extra_terms)
    if not audit["ok"]:
        raise LeakAuditError(f"Leak audit failed for {name}: {audit['violations']}")


def save_audit_report(audit: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[3/7] harden env_sanitizer.py..."
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
    抽取只允许进入 LLM 的 clean interface。

    注意：
      - 真实环境标识只在 private runtime 中使用；
      - 原始环境返回的私有评估信号不出现在这里；
      - 源码、docstring、官方奖励说明、fitness 实现都不出现在这里；
      - observation/action 只暴露形状、范围、类型，不暴露人工语义解释。
    """
    env = gym.make(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space

        return {
            "env_alias": env_alias,
            "observation_space": _space_to_public_dict(obs_space),
            "action_space": _space_to_public_dict(act_space),
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

echo "[4/7] harden reward_schema.py..."
python - <<'PY'
from pathlib import Path

path = Path("mvp/reward_schema.py")
text = path.read_text(encoding="utf-8")

old = '''        "allowed_inputs": REQUIRED_SIGNATURE,
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "components": [
'''

new = '''        "allowed_inputs": REQUIRED_SIGNATURE,
        "private_signal_policy": "Only public transition inputs are available to the reward function.",
        "components": [
'''

if old not in text:
    raise SystemExit("ERROR: expected schema block not found in mvp/reward_schema.py")

text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
PY

echo "[5/7] harden agents.py..."
cat > mvp/agents.py <<'PY'
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .leak_audit import assert_no_leak_text
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
    Bootstrap schema agent。

    只基于 PublicTaskSpec + CleanEnvInterface 生成通用 RewardSchema。
    不读取原始 env.py，不读取官方 reward，不读取 hidden evaluator 实现。
    """

    def build_schema(
        self,
        public_task: PublicTaskSpec,
        clean_interface: dict[str, Any],
    ) -> dict[str, Any]:
        return build_default_schema(public_task.__dict__, clean_interface)


class PlannerAgent:
    """
    只生成干净 plan，不暴露真实环境名，不解释 obs/action 物理含义。
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
            "Boundary:\n"
            "- Use only public observations, actions, transitions, done flag, info, and feedback summaries.\n"
            "- Do not infer the real environment identity.\n"
            "- Do not rely on private runtime signals or implementation internals."
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
            f"Reflection from previous clean candidates:\n{reflection_context}\n\n"
            f"Parent reward codes from the same clean schema only:\n{parent_block}\n\n"
            "Now generate one reward function candidate that strictly follows the schema."
        )

        assert_no_leak_text("reward_coder_user_prompt", user)

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
            "Judge visible behavior quality only if visual evidence is available.\n"
            "Do not infer the real environment identity.\n"
            "Do not use private evaluator details or generated reward magnitude as proof of success.\n"
            "Return JSON only."
        )

        assert_no_leak_text("vision_judge_user_prompt", rubric)

        out = self.model.judge_video(self.system_prompt, rubric, video_path)
        score = float(out.get("score", 0.0))
        reason = str(out.get("reason", ""))

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
                    f"private_eval_return={r.get('hidden_eval_return')}, "
                    f"generated_return={r.get('train_mean_return')}, "
                    f"reason={r.get('judge_reason')}, "
                    f"validation_errors={r.get('validation_errors', [])}"
                )
            )

        user = (
            "Past clean candidates from the same schema only:\n"
            + "\n".join(summary_lines)
            + "\n\nDo not infer the real environment identity. Propose schema-preserving mutation hypotheses."
        )

        # Reflection 允许看到私有评价数值，但不允许出现环境名和具体私有变量名。
        assert_no_leak_text(
            "reflection_user_prompt",
            user,
            extra_terms=("env_reward", "hidden_env_reward", "_hidden_env_reward", "fitness_score"),
        )

        return self.model.chat(self.system_prompt, user)
PY

echo "[6/7] harden orchestrator.py..."
cat > mvp/orchestrator.py <<'PY'
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .agents import BootstrapAgent, PlannerAgent, ReflectionAgent, RewardCoderAgent, VisionJudgeAgent
from .config import MVPConfig
from .env_sanitizer import infer_clean_env_interface
from .leak_audit import LeakAuditError, audit_text_bundle, save_audit_report
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

        audit = audit_text_bundle(
            {
                "clean_interface": clean_interface,
                "reward_schema": reward_schema,
                "clean_plan": plan,
            },
            env_id=private_task.env_id,
            extra_terms=public_task.forbidden_terms,
        )
        save_audit_report(audit, self.cfg.workspace / "leak_audit_pre_generation.json")
        if not audit["ok"]:
            raise LeakAuditError(
                "Pre-generation leak audit failed. "
                f"See {self.cfg.workspace / 'leak_audit_pre_generation.json'}"
            )

        (self.cfg.workspace / "clean_interface.txt").write_text(
            json.dumps(clean_interface, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.cfg.workspace / "reward_schema.txt").write_text(
            json.dumps(reward_schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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
                    reflection_summary=reflection,
                    reward_code=reward_code,
                    llm_rationale=rationale,
                    train_mean_return=generated_return,
                    hidden_eval_return=private_eval_return,
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
        f"selection_score_private_eval: {best.get('selection_score', 0)}",
        f"private_eval_return: {best.get('hidden_eval_return', 0)}",
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

echo "[7/7] harden prompts..."
cat > mvp/prompts/reward_coder_system.txt <<'TXT'
You are a reward-function search agent for reinforcement learning.

Generate a reward function using only the public transition interface and the provided schema. Do not infer the real environment identity. Do not rely on private runtime signals, benchmark signals, implementation internals, or private scoring logic.

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
- Design from public observations, actions, transitions, done flag, info, parent candidates, and closed-loop feedback only.
TXT

cat > mvp/prompts/reflection_system.txt <<'TXT'
You are a reflection agent for clean autonomous reward evolution.

Summarize why top clean candidates worked or failed, using only:
- schema-aligned component behavior,
- validation status,
- generated reward return as diagnostic only,
- private evaluation return as the selection metric,
- judge comments if available.

Never infer or mention the real environment identity.
Never propose using private runtime signals, benchmark signals, implementation internals, or private scoring logic.

Return:
1) What to keep
2) What to change
3) Next schema-preserving mutation hypotheses, max 5
TXT

cat > mvp/prompts/vision_judge_system.txt <<'TXT'
You are a strict visual RL policy judge.

Given anonymous task/interface evidence and rollout evidence, score visible behavior quality if visual evidence is actually available.

Do not infer or mention the real environment identity.
Do not use generated reward magnitude as proof of task success.
Do not discuss private evaluator internals.

Return JSON keys:
- score: float from 0 to 100
- reason: short string
- strengths: array of short strings
- weaknesses: array of short strings
TXT

python -m py_compile \
  mvp/leak_audit.py \
  mvp/env_sanitizer.py \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/orchestrator.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Recommended smoke test:"
echo "  python run_mvp.py --config mvp/configs/cartpole_mock.yaml --provider mock --timesteps 2000"
echo ""
echo "After run, inspect:"
echo "  cat runs/mvp*/leak_audit_pre_generation.json"
echo "  cat runs/mvp*/clean_interface.txt"
echo "  cat runs/mvp*/reward_schema.txt"
echo "  cat runs/mvp*/clean_plan.txt"
