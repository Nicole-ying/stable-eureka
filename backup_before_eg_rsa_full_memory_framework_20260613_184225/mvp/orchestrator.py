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
from .semantic_audit import audit_semantic_text_bundle, save_semantic_audit_report
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
        semantic_pre_report = audit_semantic_text_bundle(
            {
                "clean_interface": clean_interface,
                "reward_schema": reward_schema,
                "clean_plan": plan,
            }
        )
        save_semantic_audit_report(
            semantic_pre_report,
            self.cfg.workspace / "semantic_audit_pre_generation.json",
        )
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

                semantic_report = audit_semantic_text_bundle(
                    {
                        "reward_code": reward_code,
                        "llm_rationale": rationale,
                        "reflection_summary": reflection,
                    }
                )

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
                    identity_warning_count=int(semantic_report.get("identity_warning_count", 0)),
                    identity_warnings=semantic_report.get("identity_warnings", {}),
                    semantic_term_warning_count=int(semantic_report.get("semantic_term_warning_count", 0)),
                    semantic_term_warnings=semantic_report.get("semantic_term_warnings", {}),
                    semantic_warning_count=int(semantic_report.get("semantic_warning_count", 0)),
                    semantic_warnings=semantic_report.get("semantic_warnings", {}),
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
        f"identity_warning_count: {best.get('identity_warning_count', 0)}",
        f"semantic_term_warning_count: {best.get('semantic_term_warning_count', 0)}",
        f"semantic_warning_count: {best.get('semantic_warning_count', 0)}",
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
