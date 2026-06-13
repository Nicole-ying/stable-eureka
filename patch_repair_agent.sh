#!/usr/bin/env bash
set -euo pipefail

echo "[1/8] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_repair_agent_patch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/orchestrator.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/models.py \
  mvp/prompts/repair_system.txt
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/8] remove residual schema wording..."
python - <<'PY'
from pathlib import Path

path = Path("mvp/reward_schema.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    '"description": "bounded terminal success/failure shaping without reading hidden reward"',
    '"description": "bounded terminal success/failure shaping from public termination signals"',
)

path.write_text(text, encoding="utf-8")
PY

echo "[3/8] write agents.py with RepairAgent..."
cat > mvp/agents.py <<'PY'
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .leak_audit import LeakAuditError, assert_no_leak_text
from .models import ModelGateway
from .reward_schema import build_default_schema
from .task_specs import PublicTaskSpec


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
    "LunarLander",
    "BipedalWalker",
    "CartPole",
    "Acrobot",
    "MountainCar",
    "Pendulum",
)


@dataclass
class RewardDraft:
    candidate_id: str
    reward_code: str
    rationale: str


@dataclass
class RepairDraft:
    reward_code: str
    rationale: str


def _contains_private_term(text: str) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in PRIVATE_TERMS)


def _sanitize_errors(errors: list[str]) -> list[str]:
    sanitized = []
    for err in errors:
        clean = str(err)
        for term in PRIVATE_TERMS:
            clean = re.sub(re.escape(term), "[PRIVATE_TERM]", clean, flags=re.IGNORECASE)
        sanitized.append(clean)
    return sanitized


class BootstrapAgent:
    """
    Bootstrap schema agent.

    只基于 PublicTaskSpec + CleanEnvInterface 生成通用 RewardSchema。
    不读取原始环境源码，不读取专家奖励模板，不读取私有评价实现。
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


class RepairAgent:
    """
    RepairAgent 只做 schema / syntax / contract 修复。

    重要边界：
      - 如果错误代码或错误信息中出现私有词，直接拒绝 repair；
      - repair prompt 不携带私有 token；
      - repair 后仍然必须经过 validator。
    """

    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = (PROMPT_DIR / "repair_system.txt").read_text(encoding="utf-8")

    def can_repair(self, reward_code: str, validation_errors: list[str]) -> bool:
        joined_errors = "\n".join(str(e) for e in validation_errors)
        if _contains_private_term(reward_code):
            return False
        if _contains_private_term(joined_errors):
            return False
        return True

    def repair(
        self,
        reward_code: str,
        validation_errors: list[str],
        clean_interface: dict[str, Any],
        reward_schema: dict[str, Any],
        attempt_index: int,
    ) -> RepairDraft:
        if not self.can_repair(reward_code, validation_errors):
            raise LeakAuditError("Repair refused because candidate or errors contain private terms.")

        sanitized_errors = _sanitize_errors(validation_errors)

        schema_components = "\n".join(
            [
                f"- {c['id']}: {c['description']} | direction={c['direction']} | required={c['required']}"
                for c in reward_schema["components"]
            ]
        )

        user = (
            f"Repair attempt: {attempt_index}\n\n"
            f"Validation errors:\n{sanitized_errors}\n\n"
            f"Clean environment interface:\n{clean_interface}\n\n"
            f"Reward schema version: {reward_schema['schema_version']}\n"
            f"Reward signature: {reward_schema['reward_signature']}\n"
            f"Required schema components:\n{schema_components}\n\n"
            "Candidate reward code to repair:\n"
            "```python\n"
            f"{reward_code}\n"
            "```\n\n"
            "Repair only schema, syntax, numerical stability, and return-contract issues. "
            "Do not introduce any new non-public signal or infer the real environment identity."
        )

        assert_no_leak_text("repair_user_prompt", user)

        text = self.model.chat(self.system_prompt, user)
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        repaired_code = code_match.group(1).strip() if code_match else text.strip()
        rationale_match = re.search(r"RATIONALE:(.*)", text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "schema repair"
        return RepairDraft(reward_code=repaired_code, rationale=rationale)


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
                    f"repair_attempts={r.get('repair_attempts', 0)}, "
                    f"repair_success={r.get('repair_success', False)}, "
                    f"reason={r.get('judge_reason')}, "
                    f"validation_errors={r.get('validation_errors', [])}"
                )
            )

        user = (
            "Past clean candidates from the same schema only:\n"
            + "\n".join(summary_lines)
            + "\n\nDo not infer the real environment identity. Propose schema-preserving mutation hypotheses."
        )

        assert_no_leak_text(
            "reflection_user_prompt",
            user,
            extra_terms=("env_reward", "hidden_env_reward", "_hidden_env_reward", "fitness_score"),
        )

        return self.model.chat(self.system_prompt, user)
PY

echo "[4/8] write repair prompt..."
cat > mvp/prompts/repair_system.txt <<'TXT'
You are a repair agent for generated reinforcement-learning reward code.

Your job is not to design a new reward from scratch. Your job is to minimally repair a candidate so that it satisfies the provided schema and validation contract.

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
- Use only public observations, actions, transitions, done flag, and info.
- Use only Python builtins, math, and numpy symbols already available as np.
- No import statements.
- Avoid try/except, class definitions, global state, file IO, network IO, random seeds, or environment construction.
- Keep reward magnitude bounded and finite.
- Do not infer or mention the real environment identity.
- Do not introduce private runtime signals, benchmark signals, implementation internals, or private scoring logic.
TXT

echo "[5/8] write memory.py with repair metadata..."
cat > mvp/memory.py <<'PY'
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
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

        # 关键：旧泄露 memory 没有 schema_version/env_alias/status 字段，自动被过滤。
        if schema_version is not None:
            rows = [r for r in rows if r.get("schema_version") == schema_version]
        if env_alias is not None:
            rows = [r for r in rows if r.get("env_alias") == env_alias]

        rows = [r for r in rows if r.get("status") == "ok"]
        rows.sort(key=lambda r: float(r.get("selection_score", -1e18)), reverse=True)
        return rows[:k]
PY

echo "[6/8] write orchestrator.py with repair loop..."
cat > mvp/orchestrator.py <<'PY'
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .agents import (
    BootstrapAgent,
    PlannerAgent,
    ReflectionAgent,
    RepairAgent,
    RewardCoderAgent,
    VisionJudgeAgent,
)
from .config import MVPConfig
from .env_sanitizer import infer_clean_env_interface
from .leak_audit import LeakAuditError, audit_text_bundle, save_audit_report
from .memory import CandidateRecord, JsonlMemory
from .models import ModelGateway
from .reward_schema import validate_reward_code
from .rl_worker import RLWorker
from .task_specs import get_private_task_spec, get_public_task_spec, make_env_alias


MAX_REPAIR_ATTEMPTS = 2


class RewardEvolutionOrchestrator:
    def __init__(self, cfg: MVPConfig):
        self.cfg = cfg
        self.cfg.workspace.mkdir(parents=True, exist_ok=True)

        self.memory = JsonlMemory(cfg.memory_path)
        self.model = ModelGateway(cfg.model)

        self.bootstrap = BootstrapAgent()
        self.planner = PlannerAgent()
        self.coder = RewardCoderAgent(self.model)
        self.repairer = RepairAgent(self.model)
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
                validation_errors_before_repair: list[str] = []
                validation_errors_after_repair: list[str] = []
                repair_attempts = 0
                repair_success = False

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
                        validation_errors_before_repair = list(validation_errors)

                        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
                            if not self.repairer.can_repair(reward_code, validation_errors):
                                break

                            repair_attempts = attempt
                            repair_draft = self.repairer.repair(
                                reward_code=reward_code,
                                validation_errors=validation_errors,
                                clean_interface=clean_interface,
                                reward_schema=reward_schema,
                                attempt_index=attempt,
                            )

                            reward_code = repair_draft.reward_code
                            rationale = (
                                rationale
                                + f"\n\nREPAIR_ATTEMPT_{attempt}: "
                                + repair_draft.rationale
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
                    repair_attempts=repair_attempts,
                    repair_success=repair_success,
                    validation_errors_before_repair=validation_errors_before_repair,
                    validation_errors_after_repair=validation_errors_after_repair,
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
        f"repair_attempts: {best.get('repair_attempts', 0)}",
        f"repair_success: {best.get('repair_success', False)}",
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

echo "[7/8] write exporters.py with repair columns..."
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
                        "repair_attempts": row.get("repair_attempts", 0),
                        "repair_success": row.get("repair_success", False),
                        "judge_score": row.get("judge_score"),
                        "error_type": _error_type_from_reason(str(row.get("judge_reason", ""))),
                        "validation_errors": row.get("validation_errors", []),
                        "validation_errors_before_repair": row.get("validation_errors_before_repair", []),
                        "validation_errors_after_repair": row.get("validation_errors_after_repair", []),
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
        "repair_attempts",
        "repair_success",
        "judge_score",
        "error_type",
        "validation_errors",
        "validation_errors_before_repair",
        "validation_errors_after_repair",
        "judge_reason",
        "video_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv
PY

echo "[8/8] write models.py with validator-compatible mock reward..."
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
            "    act_arr = np.asarray(action, dtype=float).reshape(-1)\n"
            "    effort = float(-0.01 * np.tanh(np.linalg.norm(act_arr)))\n"
            "    terminal = float(-1.0 if done else 0.0)\n"
            "    total = progress + stability + effort + terminal\n"
            "    components = {\n"
            "        'progress': progress,\n"
            "        'stability': stability,\n"
            "        'effort': effort,\n"
            "        'terminal': terminal,\n"
            "    }\n"
            "    return float(total), components\n"
            "```\n"
            "RATIONALE: clean bounded transition reward using only public transition inputs."
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

python -m py_compile \
  mvp/reward_schema.py \
  mvp/agents.py \
  mvp/orchestrator.py \
  mvp/memory.py \
  mvp/exporters.py \
  mvp/models.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Recommended smoke test:"
echo "  python run_mvp.py --config mvp/configs/cartpole_mock.yaml --provider mock --timesteps 2000"
echo ""
echo "Inspect after run:"
echo "  cat runs/mvp*/leak_audit_pre_generation.json"
echo "  tail -n 5 runs/mvp*/memory.jsonl"
