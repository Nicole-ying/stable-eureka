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

    Validator checks legality. CandidateEvaluator checks new candidate signal.
    This gate only enforces execution boundaries. Atomic coupled packages are
    preserved as packages, especially when the package is trying to repair
    currently zero-trigger terminal event rules.
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

    TERMINAL_REPAIR_OPERATORS = {"increase_weight", "add_duration_condition", "convert_to_one_time_event"}

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
        max_edits = max(1, min(max(configured_max, plan_max), int(gate_config.get("absolute_max_edits", 8))))
        min_ratio = float(gate_config.get("min_target_ratio", 0.02))
        min_trigger = float(gate_config.get("min_target_trigger_rate", 0.01))
        atomicity = plan_metadata.get("atomicity", "separable")
        plan_type = plan_metadata.get("plan_type", "single_edit")
        is_atomic_package = atomicity == "atomic" and plan_type == "coupled_rebalancing"

        stats = diagnostic_report.get("attribution", {}).get("component_stats", {})
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for edit in edit_plan:
            ok, reason = cls._has_target_evidence(
                schema=schema,
                edit=edit,
                stats=stats,
                min_ratio=min_ratio,
                min_trigger=min_trigger,
                allow_terminal_repair=is_atomic_package,
            )
            if not ok:
                rejected = dict(edit)
                rejected["gate_reason"] = reason
                result.rejected_edits.append(rejected)
                result.warnings.append(reason)
                continue
            scored.append((cls._evidence_score(edit, stats), edit))

        if is_atomic_package:
            if len(scored) != len(edit_plan):
                result.warnings.append("Atomic package rejected because at least one edit failed the execution-boundary gate.")
                for _, edit in scored:
                    rejected = dict(edit)
                    rejected["gate_reason"] = "atomic_package_partial_gate_failure"
                    result.rejected_edits.append(rejected)
                result.accepted_edits = []
                return result
            if len(scored) <= max_edits:
                result.accepted_edits = [edit for _, edit in scored]
                result.warnings.append(f"Accepted atomic coupled package with {len(scored)} edits; gate preserved package coherence.")
                return result
            result.warnings.append(f"Atomic package had {len(scored)} edits exceeding max_edits={max_edits}; rejected whole package.")
            for _, edit in scored:
                rejected = dict(edit)
                rejected["gate_reason"] = "atomic_package_exceeds_max_edits"
                result.rejected_edits.append(rejected)
            return result

        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) > max_edits:
            result.warnings.append(f"Edit plan had {len(scored)} evidence-supported edits; kept top {max_edits}.")
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
        allow_terminal_repair: bool = False,
    ) -> Tuple[bool, str]:
        op = edit.get("operator") or edit.get("op")
        if op in {"add_component", "add_event_rule"}:
            return True, ""
        if op not in cls.TARGETED_OPERATORS:
            return True, ""
        target = edit.get("target")
        if not target:
            return False, f"Edit {op} has no target."

        component = schema.get_component(target)
        rule = schema.get_event_rule(target)
        if component is None and rule is None:
            return False, f"Target {target} does not exist in schema."

        if allow_terminal_repair and rule is not None and op in cls.TERMINAL_REPAIR_OPERATORS:
            return True, ""
        if target not in stats:
            return True, ""

        ratio = abs(float(stats.get(target, {}).get("ratio", 0.0) or 0.0))
        trigger = float(stats.get(target, {}).get("trigger_rate", 0.0) or 0.0)
        if ratio < min_ratio and trigger < min_trigger:
            return False, f"Rejected edit on low-evidence target {target}: ratio={ratio:.4f}, trigger_rate={trigger:.4f}."
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
