from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class OutcomeDecision:
    decision: str
    continuation: str
    reason: str
    task_delta: float
    hack_delta: float
    semantic_delta: float
    accepted_for_best: bool
    rollback_recommended: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "continuation": self.continuation,
            "reason": self.reason,
            "task_delta": self.task_delta,
            "hack_delta": self.hack_delta,
            "semantic_delta": self.semantic_delta,
            "accepted_for_best": self.accepted_for_best,
            "rollback_recommended": self.rollback_recommended,
        }


class OutcomeAcceptor:
    """Decide whether a measured edit outcome should be continued.

    This module does not use official/oracle reward. It accepts or rolls back
    based on internal task proxy, semantic outcome evidence, and true hack risk.
    The key distinction is that terminal one-time reward dominance with goal
    evidence should not be treated like dense shaping reward hacking.
    """

    @staticmethod
    def decide(before: Dict[str, float], after: Dict[str, float], config: Dict[str, Any] | None = None) -> OutcomeDecision:
        config = config or {}
        min_task_improvement = float(config.get("min_task_improvement", 0.02))
        max_task_drop = float(config.get("max_task_drop", 0.05))
        min_semantic_improvement = float(config.get("min_semantic_improvement", 0.05))
        max_semantic_drop = float(config.get("max_semantic_drop", 0.05))
        min_hack_improvement = float(config.get("min_hack_improvement", 0.10))
        max_hack_increase = float(config.get("max_hack_increase", 0.05))

        task_delta = float(after.get("task_score", 0.0) - before.get("task_score", 0.0))
        hack_delta = float(after.get("hack_score", 0.0) - before.get("hack_score", 0.0))
        semantic_delta = float(after.get("semantic_score", 0.0) - before.get("semantic_score", 0.0))

        terminal_goal_evidence = bool(after.get("terminal_goal_evidence", False))
        reward_repetition_risk = bool(after.get("reward_repetition_risk", False))
        high_reward_low_progress = bool(after.get("high_reward_low_progress", False))
        shaping_goal_mismatch = bool(after.get("shaping_goal_mismatch", False))
        true_hack_risk = reward_repetition_risk or high_reward_low_progress or shaping_goal_mismatch

        task_improved = task_delta >= min_task_improvement
        semantic_improved = semantic_delta >= min_semantic_improvement
        task_dropped = task_delta < -max_task_drop
        semantic_dropped = semantic_delta < -max_semantic_drop
        hack_tolerable = hack_delta <= max_hack_increase or (terminal_goal_evidence and not true_hack_risk)

        if (task_improved or semantic_improved) and hack_tolerable:
            return OutcomeDecision(
                decision="accept",
                continuation="continue_current_schema",
                reason="Internal task/semantic evidence improved and no true reward-repetition or shaping-mismatch hack risk was detected.",
                task_delta=task_delta,
                hack_delta=hack_delta,
                semantic_delta=semantic_delta,
                accepted_for_best=True,
                rollback_recommended=False,
            )

        if true_hack_risk and hack_delta > max_hack_increase:
            return OutcomeDecision(
                decision="reject",
                continuation="rollback_to_best_schema",
                reason="True semantic hack risk increased: repeated reward payment, high-reward/low-progress, or shaping-goal mismatch.",
                task_delta=task_delta,
                hack_delta=hack_delta,
                semantic_delta=semantic_delta,
                accepted_for_best=False,
                rollback_recommended=True,
            )

        if (task_dropped or semantic_dropped) and hack_delta <= -min_hack_improvement:
            return OutcomeDecision(
                decision="mixed_tradeoff",
                continuation="rollback_to_best_schema",
                reason="Hack risk improved, but internal task/semantic evidence dropped enough that this schema should not be the next base.",
                task_delta=task_delta,
                hack_delta=hack_delta,
                semantic_delta=semantic_delta,
                accepted_for_best=False,
                rollback_recommended=True,
            )

        if task_dropped and semantic_dropped:
            return OutcomeDecision(
                decision="reject",
                continuation="rollback_to_best_schema",
                reason="Both task proxy and semantic outcome evidence degraded.",
                task_delta=task_delta,
                hack_delta=hack_delta,
                semantic_delta=semantic_delta,
                accepted_for_best=False,
                rollback_recommended=True,
            )

        return OutcomeDecision(
            decision="uncertain",
            continuation="rollback_to_best_schema",
            reason="Outcome change is ambiguous under internal semantic criteria; keep as evidence but continue from best accepted schema.",
            task_delta=task_delta,
            hack_delta=hack_delta,
            semantic_delta=semantic_delta,
            accepted_for_best=False,
            rollback_recommended=True,
        )
