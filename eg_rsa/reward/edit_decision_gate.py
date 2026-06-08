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
    """Execution boundary guard for reward edits.

    Strategic reward editing is handled by the LLM agents, semantic diagnostics,
    memory, and measured outcome. This guard only checks whether an edit package
    is executable and whether an atomic package would be partially executed.
    Low step-level ratio or trigger rate is logged as a warning, not used as a
    hard rejection rule.
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
        absolute_max = int(gate_config.get("absolute_max_edits", 8))
        max_edits = max(1, min(max(configured_max, plan_max), absolute_max))

        is_atomic = plan_metadata.get("atomicity") == "atomic" and plan_metadata.get("plan_type") == "coupled_rebalancing"
        stats = diagnostic_report.get("attribution", {}).get("component_stats", {})
        semantic = diagnostic_report.get("semantic_outcome", {}) or {}

        valid_edits: List[Dict[str, Any]] = []
        for edit in edit_plan:
            ok, reason = cls._check_boundary(schema, edit)
            cls._warn_low_step_evidence(result, edit, stats, semantic)
            if ok:
                valid_edits.append(edit)
            else:
                rejected = dict(edit)
                rejected["gate_reason"] = reason
                result.rejected_edits.append(rejected)
                result.warnings.append(reason)

        if result.rejected_edits:
            if is_atomic:
                result.warnings.append("Atomic package rejected because partial execution is forbidden.")
                for edit in valid_edits:
                    rejected = dict(edit)
                    rejected["gate_reason"] = "atomic_package_partial_execution_forbidden"
                    result.rejected_edits.append(rejected)
                result.accepted_edits = []
                return result
            result.accepted_edits = valid_edits
            return result

        if len(valid_edits) > max_edits:
            result.warnings.append(f"Edit package has {len(valid_edits)} edits exceeding max_edits={max_edits}; rejected as one package.")
            for edit in valid_edits:
                rejected = dict(edit)
                rejected["gate_reason"] = "edit_count_exceeds_limit"
                result.rejected_edits.append(rejected)
            result.accepted_edits = []
            return result

        result.accepted_edits = valid_edits
        if is_atomic:
            result.warnings.append(f"Accepted atomic package with {len(valid_edits)} edits.")
        return result

    @classmethod
    def _check_boundary(cls, schema: RewardSchema, edit: Dict[str, Any]) -> Tuple[bool, str]:
        op = edit.get("operator") or edit.get("op")
        if op not in cls.TARGETED_OPERATORS:
            return True, ""
        target = edit.get("target")
        if not target:
            return False, f"Edit {op} has no target."
        if schema.get_component(target) is None and schema.get_event_rule(target) is None:
            return False, f"Target {target} does not exist in schema."
        return True, ""

    @staticmethod
    def _warn_low_step_evidence(
        result: EditDecisionGateResult,
        edit: Dict[str, Any],
        stats: Dict[str, Any],
        semantic: Dict[str, Any],
    ) -> None:
        op = edit.get("operator") or edit.get("op")
        target = edit.get("target")
        if op in {"add_component", "add_event_rule"} or not target or target not in stats:
            return
        ratio = abs(float(stats.get(target, {}).get("ratio", 0.0) or 0.0))
        trigger = float(stats.get(target, {}).get("trigger_rate", 0.0) or 0.0)
        terminal_names = set(semantic.get("terminal_rule_names", []) or [])
        terminal_goal_evidence = bool(semantic.get("terminal_goal_evidence", False))
        reward_repetition_risk = bool(semantic.get("reward_repetition_risk", False))
        if target in terminal_names and terminal_goal_evidence and not reward_repetition_risk:
            result.warnings.append(f"Terminal target {target} has low step evidence but episode semantic evidence is present; warning only.")
        elif ratio < 0.02 and trigger < 0.01:
            result.warnings.append(f"Low step evidence for target {target}: ratio={ratio:.4f}, trigger_rate={trigger:.4f}; warning only.")
