from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class RewardHackDetector:
    """Generic reward-task misalignment detectors.

    The detector avoids official/oracle rewards. It uses learned-reward
    attribution plus task-semantic rollout evidence. This distinction matters:
    a dense shaping component dominating reward is suspicious, while a one-time
    terminal success reward dominating after terminal evidence appears can be a
    sign of goal alignment rather than reward hacking.
    """

    def __init__(
        self,
        dominance_threshold: float = 0.70,
        event_toggle_threshold: int = 6,
        low_success_threshold: float = 0.20,
    ):
        self.dominance_threshold = dominance_threshold
        self.event_toggle_threshold = event_toggle_threshold
        self.low_success_threshold = low_success_threshold

    def detect(
        self,
        trajectories: Iterable[Dict[str, Any]],
        attribution: Dict[str, Any],
        semantic_outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        episodes = list(trajectories)
        semantic_outcome = semantic_outcome or {}
        flags: Dict[str, Any] = {}

        high_reward_low_progress = self._detect_high_reward_low_progress(episodes)
        raw_component_dominance = attribution.get("dominant_component_ratio", 0.0) >= self.dominance_threshold
        repeated_events = self._detect_repeated_events(episodes)
        shaping_goal_mismatch = self._detect_shaping_goal_mismatch(episodes)

        dominant = attribution.get("dominant_component")
        terminal_rule_names = set(semantic_outcome.get("terminal_rule_names", []) or [])
        terminal_goal_evidence = bool(semantic_outcome.get("terminal_goal_evidence", False))
        reward_repetition_risk = bool(semantic_outcome.get("reward_repetition_risk", False))
        one_time_repeat_violations = semantic_outcome.get("one_time_repeat_violations", []) or []

        benign_terminal_dominance = (
            raw_component_dominance
            and dominant in terminal_rule_names
            and terminal_goal_evidence
            and not reward_repetition_risk
            and not high_reward_low_progress
            and not shaping_goal_mismatch
        )
        component_dominance = raw_component_dominance and not benign_terminal_dominance

        repeated_event_exploitation = repeated_events["flag"]
        if repeated_event_exploitation and terminal_goal_evidence and not reward_repetition_risk:
            # Contact/landing toggles can indicate unstable behavior, but if the
            # reward is one-time and not repeatedly paid, this is not reward
            # exploitation by itself.
            repeated_event_exploitation = False

        flags["high_reward_low_progress"] = high_reward_low_progress
        flags["component_dominance"] = component_dominance
        flags["repeated_event_exploitation"] = repeated_event_exploitation
        flags["shaping_goal_mismatch"] = shaping_goal_mismatch
        flags["terminal_success_dominance"] = bool(benign_terminal_dominance)
        flags["unstable_contact_behavior"] = bool(semantic_outcome.get("unstable_contact_behavior", False))
        flags["reward_repetition_risk"] = reward_repetition_risk

        suspected_components: List[str] = []
        if component_dominance and dominant:
            suspected_components.append(dominant)
        if reward_repetition_risk:
            suspected_components.extend(one_time_repeat_violations)

        failure_modes: List[str] = []
        if high_reward_low_progress:
            failure_modes.append("high_reward_low_progress")
        if component_dominance:
            failure_modes.append("single_component_dominance")
        if repeated_event_exploitation:
            failure_modes.append("repeated_event_exploitation")
        if shaping_goal_mismatch:
            failure_modes.append("shaping_goal_mismatch")
        if reward_repetition_risk:
            failure_modes.append("reward_repetition_risk")

        risk_flags = [
            high_reward_low_progress,
            component_dominance,
            repeated_event_exploitation,
            shaping_goal_mismatch,
            reward_repetition_risk,
        ]
        hack_score = float(np.mean([1.0 if flag else 0.0 for flag in risk_flags])) if risk_flags else 0.0

        semantic_notes: List[str] = []
        if benign_terminal_dominance:
            semantic_notes.append("dominant_component_is_one_time_terminal_goal_with_goal_evidence")
        if flags["unstable_contact_behavior"] and not repeated_event_exploitation:
            semantic_notes.append("contact_or_landing_toggles_observed_but_not_repeated_reward_exploitation")

        return {
            "hack_flags": flags,
            "failure_modes": failure_modes,
            "hack_score": hack_score,
            "suspected_components": sorted(set(suspected_components)),
            "dominant_component": dominant,
            "dominant_component_ratio": attribution.get("dominant_component_ratio", 0.0),
            "raw_component_dominance": raw_component_dominance,
            "benign_terminal_dominance": benign_terminal_dominance,
            "repeated_event_details": repeated_events,
            "semantic_notes": semantic_notes,
        }

    @staticmethod
    def _episode_summary(episode: Dict[str, Any]) -> Tuple[float, float, float]:
        steps = episode.get("steps", [])
        reward = float(sum(step.get("reward", 0.0) for step in steps))
        if "summary" in episode:
            summary = episode["summary"]
            progress = float(
                summary.get(
                    "progress_score",
                    summary.get(
                        "landing_quality_final",
                        summary.get("approach_region_final", summary.get("progress", 0.0)),
                    ),
                )
            )
            success = float(summary.get("success", 0.0))
            return reward, progress, success

        progress_values = [RewardHackDetector._step_progress(step) for step in steps]
        success_values = [float(step.get("task_metrics", {}).get("success", 0.0)) for step in steps]
        progress = float(progress_values[-1]) if progress_values else 0.0
        success = float(np.max(success_values)) if success_values else 0.0
        return reward, progress, success

    @staticmethod
    def _step_progress(step: Dict[str, Any]) -> float:
        metrics = step.get("task_metrics", {})
        for key in ["landing_quality", "approach_and_stability", "approach_region_score", "landing_region_score", "progress"]:
            if key in metrics:
                return float(metrics.get(key, 0.0))
        return 0.0

    def _detect_high_reward_low_progress(self, episodes: List[Dict[str, Any]]) -> bool:
        if len(episodes) < 2:
            return False
        summaries = [self._episode_summary(ep) for ep in episodes]
        rewards = np.asarray([x[0] for x in summaries], dtype=float)
        progresses = np.asarray([x[1] for x in summaries], dtype=float)
        reward_med = float(np.median(rewards))
        progress_med = float(np.median(progresses))
        suspicious = (rewards >= reward_med) & (progresses <= progress_med)
        return bool(np.mean(suspicious) >= 0.4)

    def _detect_repeated_events(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        tracked_keywords = ("contact", "landing")
        event_toggle_counts: Dict[str, int] = {}
        for episode in episodes:
            last_values: Dict[str, Any] = {}
            for step in episode.get("steps", []):
                events = step.get("events", {})
                for key, value in events.items():
                    if not any(word in key for word in tracked_keywords):
                        continue
                    value = bool(value)
                    if key in last_values and bool(last_values[key]) != value:
                        event_toggle_counts[key] = event_toggle_counts.get(key, 0) + 1
                    last_values[key] = value

        if not event_toggle_counts:
            return {"flag": False, "event_toggle_counts": {}, "max_event": None, "max_toggle_count": 0}

        max_event = max(event_toggle_counts, key=event_toggle_counts.get)
        max_count = int(event_toggle_counts[max_event])
        return {
            "flag": max_count >= self.event_toggle_threshold,
            "event_toggle_counts": event_toggle_counts,
            "max_event": max_event,
            "max_toggle_count": max_count,
        }

    def _detect_shaping_goal_mismatch(self, episodes: List[Dict[str, Any]]) -> bool:
        if not episodes:
            return False
        summaries = [self._episode_summary(ep) for ep in episodes]
        rewards = np.asarray([x[0] for x in summaries], dtype=float)
        successes = np.asarray([x[2] for x in summaries], dtype=float)
        high_reward = rewards >= float(np.median(rewards))
        low_success = successes <= self.low_success_threshold
        return bool(np.mean(high_reward & low_success) >= 0.4)
