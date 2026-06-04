from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List


@dataclass
class EpisodeSummary:
    reward: float
    length: int
    success: bool = False
    final_x: float = 0.0
    start_x: float = 0.0


@dataclass
class TrainingStats:
    mean_eval_score: float
    success_rate: float
    mean_episode_length: float
    trend: str
    converged: bool
    failure_mode: str
    interrupted: bool = False
    error_message: str = ""
    reward_error_count: int = 0
    reward_last_error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "mean_eval_score": self.mean_eval_score,
            "success_rate": self.success_rate,
            "mean_episode_length": self.mean_episode_length,
            "trend": self.trend,
            "converged": self.converged,
            "failure_mode": self.failure_mode,
            "interrupted": self.interrupted,
            "error_message": self.error_message,
            "reward_error_count": self.reward_error_count,
            "reward_last_error": self.reward_last_error,
        }


def summarize_episodes(episodes: Iterable[EpisodeSummary], target_score: float) -> TrainingStats:
    data = list(episodes)
    if not data:
        return TrainingStats(0.0, 0.0, 0.0, "no_data", False, "no evaluation episodes were collected")

    rewards = [item.reward for item in data]
    lengths = [item.length for item in data]
    success_rate = sum(item.success for item in data) / len(data)
    mean_score = mean(rewards)
    mean_length = mean(lengths)
    trend = "good" if mean_score >= target_score else "needs_improvement"
    converged = mean_score >= target_score or success_rate >= 0.8
    failure_mode = "task solved; no obvious failure mode" if converged else detect_failure_mode(data, target_score)
    return TrainingStats(mean_score, success_rate, mean_length, trend, converged, failure_mode)


def detect_failure_mode(episodes: List[EpisodeSummary], target_score: float) -> str:
    mean_score = mean(item.reward for item in episodes)
    mean_length = mean(item.length for item in episodes)
    mean_delta_x = mean(item.final_x - item.start_x for item in episodes)

    if mean_length < 50:
        return "agent fails too early; strengthen survival, balance, or safety components"
    if mean_delta_x < -0.05:
        return "agent moves backward; strengthen forward-progress component"
    if mean_score < 0.3 * target_score:
        return "reward is weak or misaligned; preserve original task reward and add denser shaping"
    return "no severe failure, but reward can improve efficiency and stability"


def build_feedback(stats: TrainingStats) -> str:
    return (
        "Training diagnostic report:\n"
        f"- Mean evaluation score: {stats.mean_eval_score:.3f}\n"
        f"- Success rate: {stats.success_rate:.3f}\n"
        f"- Mean episode length: {stats.mean_episode_length:.1f}\n"
        f"- Trend: {stats.trend}\n"
        f"- Converged: {stats.converged}\n"
        f"- Main failure mode: {stats.failure_mode}\n"
        f"- Interrupted: {stats.interrupted}\n"
        f"- Reward runtime errors: {stats.reward_error_count}\n"
        "Please improve only the reward code, especially the HRDC components and stage weights."
    )


def build_scalar_feedback(stats: TrainingStats) -> str:
    return (
        f"Mean evaluation score: {stats.mean_eval_score:.3f}. "
        "Improve the reward function based only on this scalar score."
    )
