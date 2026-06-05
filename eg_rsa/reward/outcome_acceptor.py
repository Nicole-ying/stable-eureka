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
    accepted_for_best: bool
    rollback_recommended: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "continuation": self.continuation,
            "reason": self.reason,
            "task_delta": self.task_delta,
            "hack_delta": self.hack_delta,
            "accepted_for_best": self.accepted_for_best,
            "rollback_recommended": self.rollback_recommended,
        }


class OutcomeAcceptor:
    """Decide whether a measured edit outcome should be continued.

    This module stabilizes reward search. It does not use official reward and it
    does not design new rewards. It only decides whether the next search step
    should continue from the edited schema or roll back to the best accepted
    schema based on task proxy and hack-risk deltas.
    """

    @staticmethod
    def decide(before: Dict[str, float], after: Dict[str, float], config: Dict[str, Any] | None = None) -> OutcomeDecision:
        config = config or {}
        min_task_improvement = float(config.get("min_task_improvement", 0.02))
        max_task_drop = float(config.get("max_task_drop", 0.05))
        min_hack_improvement = float(config.get("min_hack_improvement", 0.10))
        max_hack_increase = float(config.get("max_hack_increase", 0.05))

        task_delta = float(after.get("task_score", 0.0) - before.get("task_score", 0.0))
        hack_delta = float(after.get("hack_score", 0.0) - before.get("hack_score", 0.0))

        if task_delta >= min_task_improvement and hack_delta <= max_hack_increase:
            return OutcomeDecision(
                decision="accept",
                continuation="continue_current_schema",
                reason="Task proxy improved enough and hack risk did not increase beyond tolerance.",
                task_delta=task_delta,
                hack_delta=hack_delta,
                accepted_for_best=True,
                rollback_recommended=False,
            )

        if task_delta < -max_task_drop and hack_delta <= -min_hack_improvement:
            return OutcomeDecision(
                decision="mixed_tradeoff",
                continuation="rollback_to_best_schema",
                reason="Hack risk improved, but task proxy dropped enough that the edited schema should not be used as the next search base.",
                task_delta=task_delta,
                hack_delta=hack_delta,
                accepted_for_best=False,
                rollback_recommended=True,
            )

        if task_delta < -max_task_drop or hack_delta > max_hack_increase:
            return OutcomeDecision(
                decision="reject",
                continuation="rollback_to_best_schema",
                reason="Outcome degraded task proxy or increased hack risk beyond tolerance.",
                task_delta=task_delta,
                hack_delta=hack_delta,
                accepted_for_best=False,
                rollback_recommended=True,
            )

        return OutcomeDecision(
            decision="uncertain",
            continuation="rollback_to_best_schema",
            reason="Outcome change is too small or ambiguous; keep lesson as evidence but continue from best accepted schema.",
            task_delta=task_delta,
            hack_delta=hack_delta,
            accepted_for_best=False,
            rollback_recommended=True,
        )
