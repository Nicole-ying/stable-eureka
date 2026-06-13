from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .agents import (
    EnvUnderstandingAgent,
    LessonExtractorAgent,
    ReflectionAgent,
    RewardCoderAgent,
    SchemaPlannerAgent,
    VisionJudgeAgent,
)
from .config import MVPConfig
from .env_sanitizer import infer_clean_env_interface
from .lessons import (
    append_jsonl,
    normalize_lesson,
    pack_candidate_evidence,
    pack_generation_evidence,
    read_jsonl,
    retrieve_memory_context,
)
from .memory import CandidateRecord, JsonlMemory
from .models import ModelGateway
from .reward_schema import validate_reward_code
from .rl_worker import RLWorker
from .task_specs import get_private_task_spec, make_env_alias


MAX_REPAIR_ATTEMPTS = 0  # RewardSpec IR should be fixed at the spec layer, not by patching Python code.


class RewardEvolutionOrchestrator:
    def __init__(self, cfg: MVPConfig):
        self.cfg = cfg
        self.cfg.workspace.mkdir(parents=True, exist_ok=True)

        self.memory = JsonlMemory(cfg.memory_path)
        self.model = ModelGateway(cfg.model)

        self.env_understander = EnvUnderstandingAgent(self.model)
        self.schema_planner = SchemaPlannerAgent(self.model)
        self.coder = RewardCoderAgent(self.model)
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
        clean_interface = infer_clean_env_interface(
            private_task.env_id,
            env_alias,
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
            parent_specs = [r.get("reward_spec", {}) for r in top[: self.cfg.memory.parent_code_top_k] if r.get("reward_spec")]
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
                reward_spec: dict = {}
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
                        parent_specs=parent_specs,
                        log_dir=candidate_llm_dir / "reward_coder",
                        parent_code_max_chars=self.cfg.memory.parent_code_max_chars,
                    )
                    reward_code = draft.reward_code
                    reward_spec = draft.reward_spec
                    rationale = draft.rationale
                    prompt_budgets["reward_spec_agent"] = draft.prompt_budget
                    prompt_paths["reward_spec_agent"] = draft.prompt_budget.get("paths", {})

                    (candidate_artifact_dir / "reward_spec.json").write_text(
                        json.dumps(reward_spec, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (candidate_artifact_dir / "reward_code.py").write_text(reward_code, encoding="utf-8")
                    (candidate_artifact_dir / "rationale.txt").write_text(rationale, encoding="utf-8")
                    artifact_paths["reward_spec"] = str(candidate_artifact_dir / "reward_spec.json")
                    artifact_paths["reward_code"] = str(candidate_artifact_dir / "reward_code.py")

                    valid, validation_errors = validate_reward_code(
                        reward_code,
                        reward_schema,
                        clean_interface,
                    )
                    validation_errors_before_repair = list(validation_errors)
                    validation_errors_after_repair = list(validation_errors)
                    self._write_json(
                        candidate_artifact_dir / "validation.json",
                        {
                            "valid": valid,
                            "validation_errors": validation_errors,
                            "validation_errors_before_repair": validation_errors_before_repair,
                            "validation_errors_after_repair": validation_errors_after_repair,
                            "reward_spec_validated": bool(reward_spec),
                            "reward_spec_id": reward_spec.get("spec_id"),
                        },
                    )

                    if not valid:
                        status = "invalid_schema"
                        judge_reason = "validation_error: " + "; ".join(validation_errors)
                    else:
                        train_result = self.worker.train_and_eval(reward_code, ckpt)
                        status = "ok"
                        self._write_json(candidate_artifact_dir / "train_result.json", train_result)

                        if self.cfg.rl.render_video:
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
                        else:
                            judge_score = 0.0
                            judge_reason = "video_render_skipped"
                            judge_details = {"render_video": False}

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
                    reward_spec=reward_spec,
                    prompt_paths=prompt_paths,
                    prompt_budgets=prompt_budgets,
                    artifact_paths=artifact_paths,
                    diagnostics=dict(train_result.get("diagnostics", {})),
                    lesson_ids=[],
                )
                as_dict = rec.__dict__

                # Candidate-level lesson extraction.
                # This gives STM a real lesson layer instead of only storing raw candidate records.
                lesson_ids: list[str] = []
                try:
                    candidate_evidence = pack_candidate_evidence(as_dict)
                    candidate_lessons_raw, candidate_lesson_budget = self.lesson_extractor.extract(
                        evidence=candidate_evidence,
                        reflection_report=rationale,
                        scope="candidate",
                        env_alias=clean_interface["env_alias"],
                        generation=g,
                        candidate_id=cid,
                        log_dir=candidate_llm_dir / "lesson_extractor_candidate",
                    )
                    candidate_lessons = [
                        normalize_lesson(
                            x,
                            scope="candidate",
                            env_alias=clean_interface["env_alias"],
                            generation=g,
                            candidate_id=cid,
                        )
                        for x in candidate_lessons_raw
                    ]
                    append_jsonl(self.cfg.candidate_lessons_path, candidate_lessons)
                    lesson_ids = [str(x.get("lesson_id")) for x in candidate_lessons]
                except Exception as e:
                    candidate_lessons = [
                        normalize_lesson(
                            {
                                "lesson_type": "extractor_error",
                                "condition": "Candidate lesson extraction failed.",
                                "observation": str(e),
                                "explanation": "Candidate lesson extraction raised an exception.",
                                "recommendation": "Inspect candidate lesson extractor prompt/response.",
                                "confidence": 0.2,
                                "reuse_policy": "same_env",
                            },
                            scope="candidate",
                            env_alias=clean_interface["env_alias"],
                            generation=g,
                            candidate_id=cid,
                        )
                    ]
                    append_jsonl(self.cfg.candidate_lessons_path, candidate_lessons)
                    lesson_ids = [str(x.get("lesson_id")) for x in candidate_lessons]

                rec.lesson_ids = lesson_ids
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
            # Quality-gate v1: do not push current-run lessons into LTM during the same run.
            # LTM should be updated by a separate cross-run promotion step, not inside each generation.

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
        "## RewardSpec JSON IR",
        "```json",
        json.dumps(best.get("reward_spec", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Prompt paths",
        "```json",
        json.dumps(best.get("prompt_paths", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Compiled reward code",
        "```python",
        best.get("reward_code", ""),
        "```",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
