from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from eg_rsa.reward.schema import RewardSchema


class SemanticOutcomeAnalyzer:
    """Aggregate task-semantic rollout evidence without using official reward.

    This analyzer is the internal alternative to using posthoc/oracle reward for
    accept/rollback decisions. It asks whether the policy shows task-aligned
    behavior according to configured diagnostic events, reward payments, and
    trajectory summaries:
      - Did success/terminal events occur in episodes?
      - Were terminal rewards paid once or repeatedly?
      - Is contact/landing behavior unstable but not necessarily reward hacking?
      - Are shaping rewards still dominating without terminal progress?
    """

    SUCCESS_KEYWORDS = ("success", "goal", "complete", "landing", "stable")

    @classmethod
    def analyze(
        cls,
        trajectories: Iterable[Dict[str, Any]],
        schema: Optional[RewardSchema] = None,
        structural_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        episodes = list(trajectories)
        n = max(1, len(episodes))
        structural_context = structural_context or {}
        schema = schema or RewardSchema(version=0, components=[], event_rules=[])

        preferred_events = set(structural_context.get("preferred_success_events", []) or [])
        terminal_rule_names = cls._terminal_rule_names(schema, preferred_events)
        one_time_rule_names = {rule.name for rule in schema.event_rules if bool(rule.one_time)}

        summaries = [ep.get("summary", {}) for ep in episodes]
        success_flags = [float(s.get("success", 0.0) or 0.0) > 0.0 for s in summaries]
        progress_values = [float(s.get("progress_score", 0.0) or 0.0) for s in summaries]
        length_values = [float(s.get("episode_length", 0.0) or 0.0) for s in summaries]
        safe_contact_flags = [float(s.get("safe_contact_rate", 0.0) or 0.0) > 0.0 for s in summaries]
        stable_landing_flags = [float(s.get("stable_landing_rate", 0.0) or 0.0) > 0.0 for s in summaries]
        contact_toggle_values = [float(s.get("max_event_toggle_count", 0.0) or 0.0) for s in summaries]

        terminal_paid_flags: List[bool] = []
        terminal_payment_counts: List[int] = []
        one_time_repeat_violations: List[str] = []
        event_rule_payment_stats: Dict[str, Dict[str, Any]] = {}
        for ep in episodes:
            steps = ep.get("steps", [])
            paid_any = False
            terminal_count = 0
            for rule_name in terminal_rule_names:
                counts = [1 for step in steps if float(step.get("components", {}).get(rule_name, 0.0) or 0.0) > 0.0]
                payment_count = len(counts)
                paid_any = paid_any or payment_count > 0
                terminal_count += payment_count
                stats = event_rule_payment_stats.setdefault(rule_name, {"episode_payment_counts": []})
                stats["episode_payment_counts"].append(payment_count)
                if rule_name in one_time_rule_names and payment_count > 1:
                    one_time_repeat_violations.append(rule_name)
            terminal_paid_flags.append(paid_any)
            terminal_payment_counts.append(terminal_count)

        for rule_name, stats in event_rule_payment_stats.items():
            counts = stats.get("episode_payment_counts", [])
            stats["episode_paid_rate"] = float(np.mean([c > 0 for c in counts])) if counts else 0.0
            stats["payment_count_mean"] = float(np.mean(counts)) if counts else 0.0
            stats["payment_count_max"] = int(max(counts)) if counts else 0
            stats["one_time"] = rule_name in one_time_rule_names

        terminal_paid_rate = float(np.mean(terminal_paid_flags)) if terminal_paid_flags else 0.0
        success_episode_rate = float(np.mean(success_flags)) if success_flags else 0.0
        stable_landing_episode_rate = float(np.mean(stable_landing_flags)) if stable_landing_flags else 0.0
        safe_contact_episode_rate = float(np.mean(safe_contact_flags)) if safe_contact_flags else 0.0
        contact_toggle_mean = float(np.mean(contact_toggle_values)) if contact_toggle_values else 0.0
        progress_mean = float(np.mean(progress_values)) if progress_values else 0.0
        episode_length_mean = float(np.mean(length_values)) if length_values else 0.0

        reward_repetition_risk = len(one_time_repeat_violations) > 0
        terminal_goal_evidence = terminal_paid_rate > 0.0 or success_episode_rate > 0.0 or stable_landing_episode_rate > 0.0
        unstable_contact_behavior = contact_toggle_mean >= 6.0

        semantic_score = (
            2.0 * success_episode_rate
            + 1.0 * terminal_paid_rate
            + 0.5 * stable_landing_episode_rate
            + progress_mean
            - 0.0001 * episode_length_mean
        )

        semantic_risks: List[str] = []
        if reward_repetition_risk:
            semantic_risks.append("one_time_reward_repeated_payment")
        if unstable_contact_behavior and not reward_repetition_risk:
            semantic_risks.append("unstable_contact_behavior")
        if terminal_goal_evidence:
            semantic_risks.append("terminal_goal_evidence_present")

        return {
            "episode_count": len(episodes),
            "semantic_score": float(semantic_score),
            "success_episode_rate": success_episode_rate,
            "terminal_reward_paid_episode_rate": terminal_paid_rate,
            "terminal_reward_payment_count_mean": float(np.mean(terminal_payment_counts)) if terminal_payment_counts else 0.0,
            "terminal_reward_payment_count_max": int(max(terminal_payment_counts)) if terminal_payment_counts else 0,
            "safe_contact_episode_rate": safe_contact_episode_rate,
            "stable_landing_episode_rate": stable_landing_episode_rate,
            "progress_score_mean": progress_mean,
            "progress_score_std": float(np.std(progress_values)) if progress_values else 0.0,
            "episode_length_mean": episode_length_mean,
            "episode_length_std": float(np.std(length_values)) if length_values else 0.0,
            "contact_toggle_mean": contact_toggle_mean,
            "contact_toggle_max": float(max(contact_toggle_values)) if contact_toggle_values else 0.0,
            "reward_repetition_risk": bool(reward_repetition_risk),
            "one_time_repeat_violations": sorted(set(one_time_repeat_violations)),
            "terminal_goal_evidence": bool(terminal_goal_evidence),
            "unstable_contact_behavior": bool(unstable_contact_behavior),
            "event_rule_payment_stats": event_rule_payment_stats,
            "terminal_rule_names": sorted(terminal_rule_names),
            "one_time_rule_names": sorted(one_time_rule_names),
            "semantic_risks": semantic_risks,
        }

    @classmethod
    def _terminal_rule_names(cls, schema: RewardSchema, preferred_events: Set[str]) -> Set[str]:
        names: Set[str] = set()
        for rule in schema.event_rules:
            condition_events = {key for key in rule.condition.keys() if key != "duration_steps"}
            name_lower = rule.name.lower()
            if condition_events & preferred_events:
                names.add(rule.name)
                continue
            if bool(rule.one_time) and any(keyword in name_lower for keyword in cls.SUCCESS_KEYWORDS):
                names.add(rule.name)
        return names
