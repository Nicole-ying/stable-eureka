from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .json_tools import write_json


class SearchController:
    """Environment-agnostic candidate acceptance and parent-selection policy.

    Important distinction:
    - A trained candidate is never trained again.
    - When a new candidate is rejected, the next LLM revision uses the current
      elite reward/schema/code as the parent input.
    - Only the newly generated child candidate is trained.

    The controller never uses task-specific thresholds such as LunarLander landing
    distances. It only compares the configured primary metric, generic behavior
    labels, and measured regressions.
    """

    def __init__(self, config: Dict[str, Any]):
        controller_cfg = config.get("controller", {}) or {}
        search_cfg = config.get("search", {}) or {}
        self.primary_metric = controller_cfg.get("primary_metric") or search_cfg.get("primary_metric") or "fitness_score"
        self.accept_if_primary_improves = bool(controller_cfg.get("accept_if_primary_improves", True))
        self.use_elite_as_parent_after_rejection = bool(
            controller_cfg.get(
                "use_elite_as_parent_after_rejection",
                controller_cfg.get("use_best_as_parent_after_rejection", True),
            )
        )
        self.allow_non_improving_parent = bool(controller_cfg.get("allow_non_improving_parent", False))
        self.reject_behavior_regression = bool(controller_cfg.get("reject_behavior_regression", True))
        self.stop_after_consecutive_rejections = int(controller_cfg.get("stop_after_consecutive_rejections", 0) or 0)

    def decide(
        self,
        parent_evidence: Dict[str, Any],
        candidate_evidence: Dict[str, Any],
        best_evidence: Dict[str, Any],
        parent_dir: str | Path,
        candidate_dir: str | Path,
        best_dir: str | Path,
        consecutive_rejections: int = 0,
    ) -> Dict[str, Any]:
        parent_score = _metric(parent_evidence, self.primary_metric)
        candidate_score = _metric(candidate_evidence, self.primary_metric)
        best_score = _metric(best_evidence, self.primary_metric)

        improves_parent = candidate_score > parent_score
        improves_best = candidate_score > best_score
        behavior_regression = self._behavior_regressed(parent_evidence, candidate_evidence)

        accepted_as_elite = bool(self.accept_if_primary_improves and improves_best)
        accepted_as_parent = bool(improves_parent or (self.allow_non_improving_parent and not behavior_regression))

        rejection_reasons = []
        if not improves_parent:
            rejection_reasons.append(
                f"primary metric '{self.primary_metric}' did not improve over parent: "
                f"candidate={candidate_score:.6g}, parent={parent_score:.6g}"
            )
        if candidate_score < best_score:
            rejection_reasons.append(
                f"candidate is below current elite on '{self.primary_metric}': "
                f"candidate={candidate_score:.6g}, elite={best_score:.6g}"
            )
        if behavior_regression and self.reject_behavior_regression:
            rejection_reasons.append("generic behavior regression detected")
            accepted_as_parent = False

        rejected = not accepted_as_parent

        next_parent_dir = str(candidate_dir)
        next_parent_source = "accepted_candidate"
        if rejected:
            next_parent_dir = str(best_dir if self.use_elite_as_parent_after_rejection else parent_dir)
            next_parent_source = "elite_parent" if self.use_elite_as_parent_after_rejection else "previous_parent"

        should_stop = False
        if self.stop_after_consecutive_rejections > 0 and rejected:
            should_stop = (consecutive_rejections + 1) >= self.stop_after_consecutive_rejections

        next_search_parent = {
            "source": next_parent_source,
            "dir": next_parent_dir,
            "uses_existing_trained_artifact_as_parent_only": True,
            "will_retrain_selected_parent": False,
            "will_train_new_child_candidate_next": True,
        }

        return {
            "file_type": "controller_decision",
            "primary_metric": self.primary_metric,
            "parent_candidate_id": parent_evidence.get("candidate_id"),
            "candidate_id": candidate_evidence.get("candidate_id"),
            "elite_candidate_id": best_evidence.get("candidate_id"),
            "scores": {
                "parent": parent_score,
                "candidate": candidate_score,
                "elite": best_score,
                "candidate_minus_parent": candidate_score - parent_score,
                "candidate_minus_elite": candidate_score - best_score,
            },
            "behavior": {
                "parent": parent_evidence.get("behavior_summary", {}),
                "candidate": candidate_evidence.get("behavior_summary", {}),
                "behavior_regression": behavior_regression,
            },
            "accepted_as_parent": accepted_as_parent,
            "accepted_as_elite": accepted_as_elite,
            "rejected": rejected,
            "rejection_reasons": rejection_reasons,
            "next_search_parent": next_search_parent,
            "rollback": {
                "enabled": rejected,
                "next_parent_source": next_parent_source,
                "next_parent_dir": next_parent_dir,
                "will_retrain_selected_parent": False,
            },
            "should_stop": should_stop,
        }

    def _behavior_regressed(self, parent: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        parent_b = parent.get("behavior_summary", {})
        cand_b = candidate.get("behavior_summary", {})
        parent_label = parent_b.get("dominant_behavior")
        cand_label = cand_b.get("dominant_behavior")
        if parent_label in {"promising_success_like_policy", "promising_policy"} and cand_label in {
            "passive_no_action_policy",
            "early_failure_or_uncontrolled_terminal",
            "hovering_or_timeout_policy",
        }:
            return True

        p_success = _behavior_rate(parent, "success_like_terminal_rate")
        c_success = _behavior_rate(candidate, "success_like_terminal_rate")
        p_len = _metric(parent, "episode_length")
        c_len = _metric(candidate, "episode_length")

        if p_success > 0.0 and c_success <= 0.0:
            return True
        if p_len > 0 and c_len > 3.0 * p_len and _metric(candidate, self.primary_metric) <= _metric(parent, self.primary_metric):
            return True
        return False


def write_controller_decision(path: str | Path, decision: Dict[str, Any]) -> None:
    write_json(path, decision)


def _metric(evidence: Dict[str, Any], key: str) -> float:
    metrics = evidence.get("primary_metrics", {}) or {}
    try:
        return float(metrics.get(key, 0.0))
    except Exception:
        return 0.0


def _behavior_rate(evidence: Dict[str, Any], key: str) -> float:
    behavior = evidence.get("behavior_summary", {}) or {}
    try:
        return float(behavior.get(key, 0.0))
    except Exception:
        return 0.0
