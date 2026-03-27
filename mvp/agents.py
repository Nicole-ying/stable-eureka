from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import ModelGateway
from .task_specs import TaskSpec


PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass
class RewardDraft:
    candidate_id: str
    reward_code: str
    rationale: str


class PlannerAgent:
    def __init__(self):
        self.system_prompt = (PROMPT_DIR / "planner_system.txt").read_text(encoding="utf-8")

    def plan(self, task: TaskSpec) -> str:
        user = (
            f"Environment: {task.env_id}\n"
            f"Objective: {task.objective}\n"
            f"Observation hints: {task.obs_hint}\n"
            f"Action hints: {task.action_hint}\n"
            f"Success patterns: {task.success_hint}\n"
            f"Failure patterns: {task.failure_hint}\n"
        )
        return user


class RewardCoderAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = (PROMPT_DIR / "reward_coder_system.txt").read_text(encoding="utf-8")

    def draft(
        self,
        candidate_id: str,
        plan: str,
        reflection_context: str,
        parent_codes: list[str],
    ) -> RewardDraft:
        parent_block = "\n\n".join(
            [f"Parent {i+1}:\n```python\n{c}\n```" for i, c in enumerate(parent_codes)]
        ) or "No parent code yet."
        user = (
            f"Candidate ID: {candidate_id}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Reflection:\n{reflection_context}\n\n"
            f"Parent reward codes:\n{parent_block}\n\n"
            "Now generate a mutated reward function candidate."
        )
        text = self.model.chat(self.system_prompt, user)
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        reward_code = code_match.group(1).strip() if code_match else text.strip()
        rationale_match = re.search(r"RATIONALE:(.*)", text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "LLM-generated mutation"
        return RewardDraft(candidate_id=candidate_id, reward_code=reward_code, rationale=rationale)


class VisionJudgeAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = (PROMPT_DIR / "vision_judge_system.txt").read_text(encoding="utf-8")

    def judge(self, task: TaskSpec, video_path, train_return: float) -> tuple[float, str, dict]:
        rubric = (
            f"Environment: {task.env_id}.\n"
            f"Objective: {task.objective}\n"
            f"Judge rubric: {task.judge_rubric}\n"
            f"Success hints: {task.success_hint}\n"
            f"Failure hints: {task.failure_hint}\n"
        )
        out = self.model.judge_video(self.system_prompt, rubric, video_path)
        score = float(out.get("score", 0.0))
        reason = str(out.get("reason", ""))
        if score <= 0:
            score = max(0.0, min(100.0, 50 + train_return / 10.0))
            reason = f"fallback_score_from_train_return={train_return:.2f}"
        return score, reason, out


class ReflectionAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = (PROMPT_DIR / "reflection_system.txt").read_text(encoding="utf-8")

    def summarize(self, top_records: list[dict]) -> str:
        if not top_records:
            return "No prior attempts. Start with dense progress shaping + stability + terminal bonuses."
        summary_lines = [
            (
                f"id={r['candidate_id']}, score={r['judge_score']:.1f}, "
                f"return={r['train_mean_return']:.2f}, reason={r['judge_reason']}, visual={r.get('judge_details', {})}"
            )
            for r in top_records
        ]
        user = "Past best candidates:\n" + "\n".join(summary_lines)
        return self.model.chat(self.system_prompt, user)
