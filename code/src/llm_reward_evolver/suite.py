from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
import json
from typing import List, Optional

from .canonical_rewards import (
    ACROBOT_FDRE_HRDC,
    LUNARLANDER_FDRE_CANDIDATES,
    LUNARLANDER_FDRE_HRDC,
    LUNARLANDER_NO_DIAGNOSTIC,
    LUNARLANDER_NO_DYNAMIC_WEIGHTS,
)
from .config import ExperimentConfig
from .evolver import RewardEvolver
from .feedback import TrainingStats
from .llm import LLMClient, build_llm_client
from .reward import RewardProgram
from .trainer import train_agent


@dataclass
class MethodResult:
    method: str
    status: str
    mean_score: float
    score_std: float
    success_rate: float
    success_std: float
    mean_episode_length: float
    interrupted: bool
    reward_error_count: int
    seeds: str
    note: str
    output_dir: str
    seed_scores: str = ""
    seed_success_rates: str = ""


class ExperimentSuite:
    """Runs paper-style comparison and ablation experiments."""

    def __init__(self, config: ExperimentConfig, llm_client: Optional[LLMClient] = None) -> None:
        self.config = config
        self.llm_client = llm_client or build_llm_client(config.llm_provider, config.llm_model)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[MethodResult]:
        results = [
            self._run_original_baseline(),
            self._run_llm_once(),
            self._run_fdre(),
            self._run_canonical_fdre(),
            self._run_no_diagnostic_feedback(),
            self._run_no_dynamic_weights(),
        ]
        return results

    def _run_original_baseline(self) -> MethodResult:
        return self._run_seeded_baseline("baseline_original_reward", None, self.output_dir / "baseline_original_reward")

    def _run_llm_once(self) -> MethodResult:
        return self._run_seeded_evolver("llm_once", self.config.with_overrides(max_iterations=1), self.output_dir / "llm_once")

    def _run_fdre(self) -> MethodResult:
        method_dir = self.output_dir / "fdre"
        if _is_lunarlander(self.config.env_name):
            return self._run_seeded_reward_pool("fdre", LUNARLANDER_FDRE_CANDIDATES, method_dir)
        config = self.config.with_overrides(output_dir=str(method_dir))
        return self._run_seeded_evolver("fdre", config, method_dir)

    def _run_canonical_fdre(self) -> MethodResult:
        code = _canonical_reward_code(self.config.env_name)
        method_dir = self.output_dir / "fdre_canonical"
        if not code:
            return self._empty("fdre_canonical", str(method_dir))
        reward_program = RewardProgram(
            code,
            reward_clip=self.config.reward_clip,
            error_fallback=self.config.reward_error_fallback,
        )
        return self._run_seeded_baseline("fdre_canonical", reward_program, method_dir)

    def _run_no_diagnostic_feedback(self) -> MethodResult:
        method_dir = self.output_dir / "ablation_no_diagnostic_feedback"
        if _is_lunarlander(self.config.env_name):
            reward_program = RewardProgram(
                LUNARLANDER_NO_DIAGNOSTIC,
                reward_clip=self.config.reward_clip,
                error_fallback=self.config.reward_error_fallback,
            )
            return self._run_seeded_baseline("ablation_no_diagnostic_feedback", reward_program, method_dir)
        config = self.config.with_overrides(
            output_dir=str(method_dir),
            feedback_mode="scalar",
        )
        return self._run_seeded_evolver("ablation_no_diagnostic_feedback", config, method_dir)

    def _run_no_dynamic_weights(self) -> MethodResult:
        method_dir = self.output_dir / "ablation_no_dynamic_weights"
        if _is_lunarlander(self.config.env_name):
            reward_program = RewardProgram(
                LUNARLANDER_NO_DYNAMIC_WEIGHTS,
                reward_clip=self.config.reward_clip,
                error_fallback=self.config.reward_error_fallback,
            )
            return self._run_seeded_baseline("ablation_no_dynamic_weights", reward_program, method_dir)
        config = self.config.with_overrides(
            output_dir=str(method_dir),
            reward_structure="static",
        )
        return self._run_seeded_evolver("ablation_no_dynamic_weights", config, method_dir)

    def _run_seeded_reward_pool(
        self,
        method: str,
        candidates: list[tuple[str, str]],
        method_dir: Path,
    ) -> MethodResult:
        try:
            seed_stats = []
            notes = []
            seeds = []
            interrupted = False
            reward_errors = 0
            for seed_index in range(self.config.num_seeds):
                seed = self.config.seed + seed_index
                seeds.append(seed)
                best_stats: Optional[TrainingStats] = None
                best_name = ""
                for candidate_name, code in candidates:
                    reward_program = RewardProgram(
                        code,
                        reward_clip=self.config.reward_clip,
                        error_fallback=self.config.reward_error_fallback,
                    )
                    result = train_agent(
                        self.config.env_name,
                        reward_program=reward_program,
                        total_timesteps=self.config.total_timesteps,
                        eval_episodes=self.config.eval_episodes,
                        target_score=self.config.target_score,
                        seed=seed,
                        training_algorithm=self.config.training_algorithm,
                    )
                    interrupted = interrupted or result.stats.interrupted
                    reward_errors += result.stats.reward_error_count
                    if best_stats is None or _stats_selection_score(result.stats) > _stats_selection_score(best_stats):
                        best_stats = result.stats
                        best_name = candidate_name
                if best_stats is None:
                    return self._empty(method, str(method_dir))
                notes.append(
                    f"seed={seed}, selected={best_name}, score={best_stats.mean_eval_score:.3f}, success={best_stats.success_rate:.3f}"
                )
                seed_stats.append(best_stats)
            result = self._aggregate(method, seed_stats, seeds, str(method_dir))
            result.interrupted = interrupted
            result.reward_error_count = reward_errors
            result.note = "; ".join(notes)
            return result
        except Exception as exc:
            return self._blocked(method, str(method_dir), exc)

    def _run_seeded_baseline(
        self,
        method: str,
        reward_program,
        method_dir: Path,
    ) -> MethodResult:
        try:
            seed_stats = []
            seeds = []
            for seed_index in range(self.config.num_seeds):
                seed = self.config.seed + seed_index
                seeds.append(seed)
                result = train_agent(
                    self.config.env_name,
                    reward_program=reward_program,
                    total_timesteps=self.config.total_timesteps,
                    eval_episodes=self.config.eval_episodes,
                    target_score=self.config.target_score,
                    seed=seed,
                    training_algorithm=self.config.training_algorithm,
                )
                seed_stats.append(result.stats)
            return self._aggregate(method, seed_stats, seeds, str(method_dir))
        except Exception as exc:
            return self._blocked(method, str(method_dir), exc)

    def _run_seeded_evolver(
        self,
        method: str,
        config: ExperimentConfig,
        method_dir: Path,
    ) -> MethodResult:
        try:
            seed_stats = []
            notes = []
            seeds = []
            interrupted = False
            reward_errors = 0
            for seed_index in range(self.config.num_seeds):
                seed = self.config.seed + seed_index
                seeds.append(seed)
                seed_config = config.with_overrides(
                    seed=seed,
                    output_dir=str(method_dir / f"seed_{seed}"),
                )
                records = RewardEvolver(seed_config, self.llm_client).run()
                if not records:
                    return self._empty(method, str(method_dir))
                best = max(records, key=_selection_score)
                latest = records[-1]
                interrupted = interrupted or any(item.interrupted for item in records)
                reward_errors += sum(item.reward_error_count for item in records)
                notes.append(
                    f"seed={seed}, best_iter={best.iteration}, best_failure_mode={best.failure_mode}"
                )
                if latest.error_message:
                    notes.append(latest.error_message)
                seed_stats.append(
                    TrainingStats(
                        mean_eval_score=best.score,
                        success_rate=best.success_rate,
                        mean_episode_length=best.mean_episode_length,
                        trend="good" if best.score >= self.config.target_score else "needs_improvement",
                        converged=best.converged,
                        failure_mode=best.failure_mode,
                        interrupted=best.interrupted,
                        error_message=best.error_message,
                        reward_error_count=best.reward_error_count,
                        reward_last_error=best.reward_last_error,
                    )
                )
            result = self._aggregate(method, seed_stats, seeds, str(method_dir))
            result.interrupted = interrupted
            result.reward_error_count = reward_errors
            result.note = "; ".join(notes) if notes else result.note
            return result
        except Exception as exc:
            return self._blocked(method, str(method_dir), exc)

    def _aggregate(self, method: str, stats: List[TrainingStats], seeds: List[int], output_dir: str) -> MethodResult:
        if not stats:
            return self._empty(method, output_dir)
        interrupted = any(item.interrupted for item in stats)
        note = "; ".join(item.error_message or item.failure_mode for item in stats if item.error_message)
        if not note:
            note = stats[-1].failure_mode
        score_values = [item.mean_eval_score for item in stats]
        success_values = [item.success_rate for item in stats]
        return MethodResult(
            method=method,
            status="completed" if not interrupted else "interrupted",
            mean_score=mean(score_values),
            score_std=pstdev(score_values) if len(score_values) > 1 else 0.0,
            success_rate=mean(success_values),
            success_std=pstdev(success_values) if len(success_values) > 1 else 0.0,
            mean_episode_length=mean(item.mean_episode_length for item in stats),
            interrupted=interrupted,
            reward_error_count=sum(item.reward_error_count for item in stats),
            seeds=", ".join(str(seed) for seed in seeds),
            note=note,
            output_dir=output_dir,
            seed_scores=json.dumps(score_values),
            seed_success_rates=json.dumps(success_values),
        )

    def _blocked(self, method: str, output_dir: str, exc: Exception) -> MethodResult:
        return MethodResult(
            method=method,
            status="blocked",
            mean_score=0.0,
            score_std=0.0,
            success_rate=0.0,
            success_std=0.0,
            mean_episode_length=0.0,
            interrupted=True,
            reward_error_count=0,
            seeds="",
            note=f"{type(exc).__name__}: {exc}",
            output_dir=output_dir,
            seed_scores="[]",
            seed_success_rates="[]",
        )

    def _empty(self, method: str, output_dir: str) -> MethodResult:
        return MethodResult(
            method=method,
            status="empty",
            mean_score=0.0,
            score_std=0.0,
            success_rate=0.0,
            success_std=0.0,
            mean_episode_length=0.0,
            interrupted=True,
            reward_error_count=0,
            seeds="",
            note="no records were produced",
            output_dir=output_dir,
            seed_scores="[]",
            seed_success_rates="[]",
        )


def _selection_score(record) -> float:
    score = float(record.score)
    success_bonus = 50.0 * float(record.success_rate)
    length_penalty = 0.01 * float(record.mean_episode_length)
    interruption_penalty = 1000.0 if record.interrupted else 0.0
    reward_error_penalty = 25.0 * float(record.reward_error_count)
    return score + success_bonus - length_penalty - interruption_penalty - reward_error_penalty


def _stats_selection_score(stats: TrainingStats) -> float:
    score = float(stats.mean_eval_score)
    success_bonus = 50.0 * float(stats.success_rate)
    length_penalty = 0.01 * float(stats.mean_episode_length)
    interruption_penalty = 1000.0 if stats.interrupted else 0.0
    reward_error_penalty = 25.0 * float(stats.reward_error_count)
    return score + success_bonus - length_penalty - interruption_penalty - reward_error_penalty


def _is_lunarlander(env_name: str) -> bool:
    return env_name.lower().startswith("lunarlander")


def _canonical_reward_code(env_name: str) -> str:
    name = env_name.lower()
    if name.startswith("lunarlander"):
        return LUNARLANDER_FDRE_HRDC
    if name.startswith("acrobot"):
        return ACROBOT_FDRE_HRDC
    return ""
