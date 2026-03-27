from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .agents import PlannerAgent, ReflectionAgent, RewardCoderAgent, VisionJudgeAgent
from .config import MVPConfig
from .memory import CandidateRecord, JsonlMemory
from .models import ModelGateway
from .rl_worker import RLWorker
from .task_specs import get_task_spec


class RewardEvolutionOrchestrator:
    def __init__(self, cfg: MVPConfig):
        self.cfg = cfg
        self.cfg.workspace.mkdir(parents=True, exist_ok=True)
        self.memory = JsonlMemory(cfg.memory_path)
        self.model = ModelGateway(cfg.model)
        self.planner = PlannerAgent()
        self.coder = RewardCoderAgent(self.model)
        self.judge = VisionJudgeAgent(self.model)
        self.reflector = ReflectionAgent(self.model)
        self.worker = RLWorker(cfg.rl)

    def run(self) -> dict:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        task = get_task_spec(self.cfg.rl.env_id)
        plan = self.planner.plan(task)
        best: dict | None = None
        stagnant = 0
        best_score = float("-inf")

        for g in range(self.cfg.evolution.generations):
            top = self.memory.top_candidates(self.cfg.evolution.reflection_top_k)
            parent_codes = [r["reward_code"] for r in top]
            parent_ids = [r["candidate_id"] for r in top]
            try:
                reflection = self.reflector.summarize(top)
            except Exception as e:
                reflection = f"reflection_error: {e}"

            generation_best = float("-inf")
            for i in range(self.cfg.evolution.population_size):
                cid = f"g{g}_c{i}"
                ckpt = self.cfg.checkpoints_dir / f"{cid}.zip"
                video = self.cfg.videos_dir / f"{cid}.gif"

                try:
                    draft = self.coder.draft(cid, plan, reflection, parent_codes)
                    train_ret = self.worker.train_and_eval(draft.reward_code, ckpt)
                    self.worker.render_rollout_video(ckpt, video)
                    judge_score, judge_reason, judge_details = self.judge.judge(task, video, train_ret)
                    reward_code = draft.reward_code
                    rationale = draft.rationale
                except Exception as e:  # resilient closed-loop search
                    train_ret = -1e9
                    judge_score = 0.0
                    judge_reason = f"pipeline_error: {e}"
                    reward_code = ""
                    rationale = "pipeline failed before reward candidate became runnable"
                    judge_details = {"error": str(e)}

                rec = CandidateRecord(
                    generation=g,
                    candidate_id=cid,
                    parent_ids=parent_ids,
                    reflection_summary=reflection,
                    reward_code=reward_code,
                    llm_rationale=rationale,
                    train_mean_return=train_ret,
                    judge_score=judge_score,
                    judge_reason=judge_reason,
                    judge_details=judge_details,
                    video_path=str(video),
                )
                self.memory.append(rec)
                as_dict = rec.__dict__

                generation_best = max(generation_best, judge_score)
                if best is None or as_dict["judge_score"] > best["judge_score"]:
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
        "# MVP Run Report",
        f"best_candidate: {best.get('candidate_id', 'N/A')}",
        f"judge_score: {best.get('judge_score', 0)}",
        f"train_mean_return: {best.get('train_mean_return', 0)}",
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
