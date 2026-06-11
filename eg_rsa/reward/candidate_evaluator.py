from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from eg_rsa.reward.operators import RewardEditOperatorApplier


@dataclass
class CandidateEvaluationResult:
    accepted_edits: List[Dict[str, Any]] = field(default_factory=list)
    rejected_edits: List[Dict[str, Any]] = field(default_factory=list)
    reports: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_edits": self.accepted_edits,
            "rejected_edits": self.rejected_edits,
            "reports": self.reports,
            "warnings": self.warnings,
        }


class RewardCandidateEvaluator:
    """Evaluate whether new structural reward candidates are learnable.

    This module is intentionally environment-agnostic. It uses only recorded
    rollout events and task_metrics to estimate whether a proposed new reward
    would have produced any signal under the current policy distribution.
    """

    STRUCTURAL_OPERATORS = {"add_event_rule", "add_component"}

    @classmethod
    def evaluate(
        cls,
        edit_plan: List[Dict[str, Any]],
        trajectories: List[Dict[str, Any]],
        config: Dict[str, Any] | None = None,
    ) -> CandidateEvaluationResult:
        config = config or {}
        min_event_trigger_rate = float(config.get("min_event_trigger_rate", 0.001))
        min_metric_variation = float(config.get("min_metric_variation", 1e-4))
        min_metric_active_rate = float(config.get("min_metric_active_rate", 0.01))
        reject_zero_signal = bool(config.get("reject_zero_signal", True))
        hard_filter = bool(config.get("hard_filter", False))

        result = CandidateEvaluationResult()
        for edit in edit_plan:
            op = edit.get("operator") or edit.get("op")
            if op not in cls.STRUCTURAL_OPERATORS:
                result.accepted_edits.append(edit)
                continue

            report = cls._evaluate_one(edit, trajectories)
            result.reports.append(report)
            recommendation = report.get("recommendation", "accept")

            # V2 slim pipeline:
            # Candidate evaluation is a feedback signal by default. It should not
            # block formula-native reward self-evolution unless config explicitly
            # requests hard_filter=true.
            if not hard_filter:
                if recommendation not in {"accept", "", None}:
                    result.warnings.append(
                        "candidate_evaluator advisory_only: "
                        + str(report.get("reason", recommendation))
                    )
                result.accepted_edits.append(edit)
                continue

            if reject_zero_signal and recommendation == "reject_as_too_sparse":
                rejected = dict(edit)
                rejected["candidate_eval_reason"] = report.get("reason")
                result.rejected_edits.append(rejected)
                result.warnings.append(str(report.get("reason")))
                continue
            if recommendation == "prefer_process_signal":
                rejected = dict(edit)
                rejected["candidate_eval_reason"] = report.get("reason")
                result.rejected_edits.append(rejected)
                result.warnings.append(str(report.get("reason")))
                continue
            if report.get("candidate_type") == "metric_process":
                variation = float(report.get("metric_variation", 0.0) or 0.0)
                active_rate = float(report.get("active_rate", 0.0) or 0.0)
                if variation < min_metric_variation and active_rate < min_metric_active_rate:
                    rejected = dict(edit)
                    rejected["candidate_eval_reason"] = report.get("reason")
                    result.rejected_edits.append(rejected)
                    result.warnings.append(str(report.get("reason")))
                    continue
            if report.get("candidate_type") in {"terminal_event", "intermediate_event"}:
                trigger_rate = float(report.get("estimated_trigger_rate", 0.0) or 0.0)
                if trigger_rate < min_event_trigger_rate:
                    rejected = dict(edit)
                    rejected["candidate_eval_reason"] = report.get("reason")
                    result.rejected_edits.append(rejected)
                    result.warnings.append(str(report.get("reason")))
                    continue
            result.accepted_edits.append(edit)
        return result

    @classmethod
    def _evaluate_one(cls, edit: Dict[str, Any], trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        op = edit.get("operator") or edit.get("op")
        if op == "add_event_rule":
            return cls._evaluate_event_rule(edit, trajectories)
        if op == "add_component":
            component = edit.get("component", {})
            component_type = component.get("type")
            if component_type in RewardEditOperatorApplier.METRIC_COMPONENT_TYPES:
                return cls._evaluate_metric_component(edit, trajectories)
        return {
            "candidate_type": "structural_unknown",
            "recommendation": "accept",
            "reason": "No candidate feasibility check for this structural edit type.",
        }

    @classmethod
    def _evaluate_event_rule(cls, edit: Dict[str, Any], trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        rule = edit.get("event_rule", {})
        condition = rule.get("condition", {}) if isinstance(rule, dict) else {}
        trigger_count, total_steps, episode_trigger_count = cls._event_condition_counts(condition, trajectories)
        trigger_rate = trigger_count / max(1, total_steps)
        episode_rate = episode_trigger_count / max(1, len(trajectories))
        event_keys = [k for k in condition.keys() if k != "duration_steps"]
        candidate_type = "terminal_event" if len(event_keys) >= 3 or bool(rule.get("one_time", False)) else "intermediate_event"
        signal_density = cls._density_from_rate(trigger_rate)
        if trigger_count == 0:
            return {
                "candidate_type": candidate_type,
                "estimated_trigger_rate": 0.0,
                "episode_trigger_rate": 0.0,
                "signal_density": "zero",
                "recommendation": "prefer_process_signal",
                "reason": f"New event rule {rule.get('name')} would not trigger on current rollouts; prefer metric-based process signal before training this sparse event.",
            }
        return {
            "candidate_type": candidate_type,
            "estimated_trigger_rate": float(trigger_rate),
            "episode_trigger_rate": float(episode_rate),
            "signal_density": signal_density,
            "recommendation": "accept",
            "reason": f"Event rule {rule.get('name')} has non-zero estimated trigger signal on current rollouts.",
        }

    @classmethod
    def _evaluate_metric_component(cls, edit: Dict[str, Any], trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        component = edit.get("component", {})
        params = component.get("params", {}) if isinstance(component, dict) else {}
        metric = params.get("metric")
        component_type = component.get("type")
        values = cls._metric_values(metric, trajectories)
        if not values:
            return {
                "candidate_type": "metric_process",
                "metric": metric,
                "component_type": component_type,
                "metric_variation": 0.0,
                "active_rate": 0.0,
                "signal_density": "zero",
                "recommendation": "reject_as_too_sparse",
                "reason": f"Metric component references metric {metric!r}, but this metric is absent in current rollouts.",
            }
        variation = max(values) - min(values)
        if component_type == "metric_delta":
            positives = 0
            comparisons = 0
            last = None
            for value in values:
                if last is not None:
                    comparisons += 1
                    if value - last > 0:
                        positives += 1
                last = value
            active_rate = positives / max(1, comparisons)
        elif component_type == "metric_threshold_bonus":
            threshold = float(params.get("threshold", 0.0))
            direction = params.get("direction", "ge")
            if direction == "le":
                active_rate = sum(1 for v in values if v <= threshold) / max(1, len(values))
            else:
                active_rate = sum(1 for v in values if v >= threshold) / max(1, len(values))
        elif component_type == "metric_stagnation_penalty":
            threshold = float(params.get("threshold", 1e-3))
            stagnant = 0
            comparisons = 0
            last = None
            for value in values:
                if last is not None:
                    comparisons += 1
                    if abs(value - last) < threshold:
                        stagnant += 1
                last = value
            active_rate = stagnant / max(1, comparisons)
        else:
            active_rate = sum(1 for v in values if abs(v) > 0.0) / max(1, len(values))
        signal_density = cls._density_from_rate(active_rate)
        recommendation = "accept" if variation > 0.0 or active_rate > 0.0 else "reject_as_too_sparse"
        return {
            "candidate_type": "metric_process",
            "metric": metric,
            "component_type": component_type,
            "metric_variation": float(variation),
            "active_rate": float(active_rate),
            "signal_density": signal_density,
            "recommendation": recommendation,
            "reason": f"Metric component {component.get('name')} has variation={variation:.6f}, active_rate={active_rate:.6f} on current rollouts.",
        }

    @staticmethod
    def _event_condition_counts(condition: Dict[str, Any], trajectories: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        trigger_count = 0
        total_steps = 0
        episode_trigger_count = 0
        duration_steps = int(condition.get("duration_steps", 1) or 1)
        event_keys = [k for k in condition.keys() if k != "duration_steps"]
        for traj in trajectories:
            consecutive = 0
            episode_triggered = False
            for step in traj.get("steps", []):
                total_steps += 1
                events = step.get("events", {})
                ok = all(bool(events.get(key, False)) == bool(condition.get(key)) for key in event_keys)
                if ok:
                    consecutive += 1
                else:
                    consecutive = 0
                if ok and consecutive >= duration_steps:
                    trigger_count += 1
                    episode_triggered = True
            if episode_triggered:
                episode_trigger_count += 1
        return trigger_count, total_steps, episode_trigger_count

    @staticmethod
    def _metric_values(metric: str, trajectories: List[Dict[str, Any]]) -> List[float]:
        values: List[float] = []
        if not metric:
            return values
        for traj in trajectories:
            for step in traj.get("steps", []):
                metrics = step.get("task_metrics", {})
                if metric in metrics:
                    try:
                        values.append(float(metrics[metric]))
                    except (TypeError, ValueError):
                        continue
        return values

    @staticmethod
    def _density_from_rate(rate: float) -> str:
        if rate <= 0.0:
            return "zero"
        if rate < 0.01:
            return "sparse"
        if rate < 0.10:
            return "medium"
        return "dense"
