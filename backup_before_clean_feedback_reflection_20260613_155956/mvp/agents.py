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



def _extract_code_and_rationale(text: str, stage: str) -> tuple[str, str]:
    """
    Robustly extract reward code from LLM output.

    支持：
      - ```python ... ```
      - ```py ... ```
      - ``` ... ```
      - 直接从 def compute_reward(...) 开始的裸代码

    如果提取不到 compute_reward，直接报错，不再让空代码进入 validator。
    """
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
        reward_code, rationale = _extract_code_and_rationale(text, stage="reward_generation")
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
        repaired_code, rationale = _extract_code_and_rationale(text, stage="repair")
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
