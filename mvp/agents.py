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
