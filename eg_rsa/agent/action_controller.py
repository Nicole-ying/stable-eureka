from __future__ import annotations

from typing import Any, Dict, List

from eg_rsa.agent.agent_action import AgentActionDecision


class AgentActionController:
    """Rule-backed v1 controller.

    This is the first executable version of the AgentActionController.
    Later it can be replaced or augmented by an LLM controller, but this
    version already changes the architecture from "always edit" to
    "choose an action based on state".
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def decide(
        self,
        diagnostic_report: Dict[str, Any],
        semantic_outcome: Dict[str, Any],
        retrieved_lessons: List[Dict[str, Any]] | None = None,
        current_phase: str | None = None,
    ) -> AgentActionDecision:
        retrieved_lessons = retrieved_lessons or []
        phase = current_phase or self.detect_phase(diagnostic_report, semantic_outcome)
        diagnostics = diagnostic_report.get("diagnostics", {}) or {}
        flags = diagnostics.get("hack_flags", {}) or {}

        success_rate = float(semantic_outcome.get("success_episode_rate", 0.0) or 0.0)
        terminal_rate = float(semantic_outcome.get("terminal_reward_paid_episode_rate", 0.0) or 0.0)
        hack_score = float(diagnostics.get("hack_score", 0.0) or 0.0)
        reward_repetition = bool(semantic_outcome.get("reward_repetition_risk", False))
        high_reward_low_progress = bool(flags.get("high_reward_low_progress", False))
        shaping_mismatch = bool(flags.get("shaping_goal_mismatch", False))

        if reward_repetition or high_reward_low_progress:
            return AgentActionDecision.run_tool(
                tool_name="schema_diff",
                args={"mode": "inspect_last_transition"},
                reason="True reward-risk signal is present; inspect transition before editing.",
            )

        if success_rate >= 0.7 and hack_score <= 0.05:
            return AgentActionDecision(
                action="continue_training",
                confidence=0.75,
                reason_summary="Reward appears aligned; prefer checkpoint continuation before risky reward edits.",
                safety_requirements=["avoid_large_structural_edit", "use_scale_audit_before_new_penalty"],
                metadata={"phase": phase, "success_rate": success_rate},
            )

        if terminal_rate > 0.0 and success_rate < 0.7:
            return AgentActionDecision(
                action="apply_local_edit",
                confidence=0.65,
                reason_summary="Terminal evidence exists but success is not stable; use conservative local refinement.",
                safety_requirements=["atomic_package_if_rebalancing", "no_large_new_dense_penalty"],
                metadata={"phase": phase, "terminal_rate": terminal_rate},
            )

        if shaping_mismatch or hack_score > 0.0:
            return AgentActionDecision(
                action="apply_local_edit",
                confidence=0.7,
                reason_summary="Reward attribution suggests shaping mismatch; use constrained local edit first.",
                safety_requirements=["scale_audit_if_add_component"],
                metadata={"phase": phase, "hack_score": hack_score},
            )

        if retrieved_lessons:
            return AgentActionDecision(
                action="run_tool",
                confidence=0.55,
                reason_summary="No strong edit signal; inspect memory and trajectories before deciding.",
                tools_to_call=[
                    {"tool_name": "trajectory_inspector", "args": {"mode": "summary"}},
                    {"tool_name": "memory_retriever", "args": {"top_k": 5}},
                ],
                metadata={"phase": phase},
            )

        return AgentActionDecision.continue_training(
            reason="No strong reward-edit signal; continue policy optimization or gather more evidence.",
            confidence=0.5,
        )

    def detect_phase(self, diagnostic_report: Dict[str, Any], semantic_outcome: Dict[str, Any]) -> str:
        diagnostics = diagnostic_report.get("diagnostics", {}) or {}
        success_rate = float(semantic_outcome.get("success_episode_rate", 0.0) or 0.0)
        terminal_rate = float(semantic_outcome.get("terminal_reward_paid_episode_rate", 0.0) or 0.0)
        hack_score = float(diagnostics.get("hack_score", 0.0) or 0.0)

        if success_rate >= 0.7 and hack_score <= 0.05:
            return "refinement"
        if terminal_rate > 0.0 or success_rate > 0.0:
            return "transition"
        return "constrained_search"
