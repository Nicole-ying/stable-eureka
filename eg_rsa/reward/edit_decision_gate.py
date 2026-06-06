from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from eg_rsa.reward.schema import RewardSchema


@dataclass
class EditDecisionGateResult:
    accepted_edits: List[Dict[str, Any]] = field(default_factory=list)
    rejected_edits: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    plan_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_edits": self.accepted_edits,
            "rejected_edits": self.rejected_edits,
            "warnings": self.warnings,
            "plan_metadata": self.plan_metadata,
        }


class EditDecisionGate:
    """Plan-level evidence gate for reward edits.

    The validator checks syntactic legality. This gate checks evidence and
    execution boundaries. It should not destroy an LLM's explicitly atomic
    coupled rebalancing plan by selecting only the top single edit.
    """

    TARGETED_OPERATORS = {
        "increase_weight",
        "decrease_weight",
        "clip_component",
        "disable_component",
        "convert_to_one_time_event",
        "add_duration_condition",
        "reshape_sparse_to_dense",
    }

    @classmethod
    def apply(
        cls,
        schema: RewardSchema,
        edit_plan: List[Dict[str, Any]],
        diagnostic_report: Dict[str, Any],
        gate_config: Dict[str, Any] | None = None,
        plan_metadata: Dict[str, Any] | None = None,
    ) -> EditDecisionGateResult:
        gate_config = gate_config or {}
        plan_metadata = plan_metadata or {}
        result = EditDecisionGateResult(plan_metadata=dict(plan_metadata))
        configured_max = int(gate_config.get("max_edits_per_iteration", 1))
        plan_max = int(plan_metadata.get("max_reasonable_edits", configured_max) or configured_max)
        max_edits = max(1, min(max(configured_max, plan_max), int(gate_config.get("absolute_max_edits", 6))))
        min_ratio = float(gate_config.get("min_target_ratio", 0.02))
        min_trigger = float(gate_config.get("min_target_trigger_rate", 0.01))
        atomicity = plan_metadata.get("atomicity", "separable")
        plan_type = plan_metadata.get("plan_type", "single_edit")

        stats = diagnostic_report.get("attribution", {}).get("component_stats", {})
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for edit in edit_plan:
            ok, reason = cls._has_target_evidence(schema, edit, stats, min_ratio, min_trigger)
            if not ok:
                rejected = dict(edit)
                rejected["gate_reason"] = reason
                result.rejected_edits.append(rejected)
                result.warnings.append(reason)
                continue
            scored.append((cls._evidence_score(edit, stats), edit))

        if atomicity == "atomic" and plan_type == "coupled_rebalancing":
            if len(scored) <= max_edits:
                result.accepted_edits = [edit for _, edit in scored]
                if len(scored) > 1:
                    result.warnings.append(
                        f"Accepted atomic coupled package with {len(scored)} edits; gate preserved package coherence."
                    )
                return result
            result.warnings.append(
                f"Atomic package had {len(scored)} edits exceeding max_edits={max_edits}; rejected whole package to avoid partial execution."
            )
            for _, edit in scored:
                rejected = dict(edit)
                rejected["gate_reason"] = "atomic_package_exceeds_max_edits"
                result.rejected_edits.append(rejected)
            return result

        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) > max_edits:
            result.warnings.append(
                f"Edit plan had {len(scored)} evidence-supported edits; kept top {max_edits}."
            )
        for idx, (_, edit) in enumerate(scored):
            if idx < max_edits:
                result.accepted_edits.append(edit)
            else:
                rejected = dict(edit)
                rejected["gate_reason"] = "multi_edit_limited_for_attribution"
                result.rejected_edits.append(rejected)
        return result

    @classmethod
    def _has_target_evidence(
        cls,
        schema: RewardSchema,
        edit: Dict[str, Any],
        stats: Dict[str, Any],
        min_ratio: float,
        min_trigger: float,
    ) -> Tuple[bool, str]:
        op = edit.get("operator") or edit.get("op")
        if op in {"add_component", "add_event_rule"}:
            return True, ""
        if op not in cls.TARGETED_OPERATORS:
            return True, ""
        target = edit.get("target")
        if not target:
            return False, f"Edit {op} has no target."

        item_exists = schema.get_component(target) is not None or schema.get_event_rule(target) is not None
        if not item_exists:
            return False, f"Target {target} does not exist in schema."
        if target not in stats:
            return True, ""

        ratio = abs(float(stats.get(target, {}).get("ratio", 0.0) or 0.0))
        trigger = float(stats.get(target, {}).get("trigger_rate", 0.0) or 0.0)
        if ratio < min_ratio and trigger < min_trigger:
            return False, (
                f"Rejected edit on low-evidence target {target}: ratio={ratio:.4f}, "
                f"trigger_rate={trigger:.4f}."
            )
        return True, ""

    @staticmethod
    def _evidence_score(edit: Dict[str, Any], stats: Dict[str, Any]) -> float:
        op = edit.get("operator") or edit.get("op")
        if op in {"add_component", "add_event_rule"}:
            return 0.5
        target = edit.get("target")
        if not target or target not in stats:
            return 0.0
        values = stats.get(target, {})
        ratio = abs(float(values.get("ratio", 0.0) or 0.0))
        trigger = float(values.get("trigger_rate", 0.0) or 0.0)
        return ratio + 0.1 * trigger
