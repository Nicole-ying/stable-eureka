from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eg_rsa.reward.operators import RewardEditOperatorApplier
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.reward.schema_canonicalizer import SchemaCanonicalizer


@dataclass
class SchemaTransitionResult:
    """Single source of truth for whether an edit actually changes schema.

    Important distinction:
      - candidate_result.accepted_edits: passed candidate feasibility
      - gate_result.accepted_edits: passed edit gate
      - committed_edits: actually applied to the next RewardSchema

    Only committed_edits are allowed to change schema.
    """

    decision: str
    next_action: str
    requested_next_action: str
    schema_changed: bool
    committed_edits: List[Dict[str, Any]] = field(default_factory=list)
    rejected_edits: List[Dict[str, Any]] = field(default_factory=list)
    blocked_candidate_edits: List[Dict[str, Any]] = field(default_factory=list)
    blocked_gate_edits: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    trace: Dict[str, Any] = field(default_factory=dict)
    next_schema: Optional[RewardSchema] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "next_action": self.next_action,
            "requested_next_action": self.requested_next_action,
            "schema_changed": self.schema_changed,
            "committed_edits": self.committed_edits,
            "rejected_edits": self.rejected_edits,
            "blocked_candidate_edits": self.blocked_candidate_edits,
            "blocked_gate_edits": self.blocked_gate_edits,
            "reason": self.reason,
            "trace": self.trace,
            "next_schema_version": self.next_schema.version if self.next_schema is not None else None,
        }


class SchemaTransitionEngine:
    """Resolve one EG-RSA iteration into a transactional schema transition.

    Runner should not directly interpret edit_plan / candidate_result /
    gate_result / next_action. It should call this engine and commit only the
    returned committed_edits.
    """

    @classmethod
    def resolve(
        cls,
        schema: RewardSchema,
        edit_plan: List[Dict[str, Any]],
        validation: Any,
        candidate_result: Any = None,
        gate_result: Any = None,
        next_action: str = "apply_edit",
        should_edit: bool = True,
        operator_constraints_enabled: bool = True,
        primitive_interface: Optional[Dict[str, Any]] = None,
    ) -> SchemaTransitionResult:
        requested_next_action = str(next_action or "apply_edit")
        validation_errors = cls._get_list(validation, "errors")
        validation_warnings = cls._get_list(validation, "warnings")

        candidate_accepted = cls._get_list(candidate_result, "accepted_edits")
        candidate_rejected = cls._get_list(candidate_result, "rejected_edits")
        gate_accepted = cls._get_list(gate_result, "accepted_edits")
        gate_rejected = cls._get_list(gate_result, "rejected_edits")

        final_plan = copy.deepcopy(edit_plan or [])

        trace = {
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
            "candidate_accepted_count": len(candidate_accepted),
            "candidate_rejected_count": len(candidate_rejected),
            "gate_accepted_count": len(gate_accepted),
            "gate_rejected_count": len(gate_rejected),
            "final_edit_plan_count": len(final_plan),
            "operator_constraints_enabled": bool(operator_constraints_enabled),
            "source_of_truth": "committed_edits",
            "notes": [],
        }

        if not should_edit or requested_next_action == "final_iteration":
            return SchemaTransitionResult(
                decision="final",
                next_action="final_iteration",
                requested_next_action=requested_next_action,
                schema_changed=False,
                committed_edits=[],
                rejected_edits=[],
                blocked_candidate_edits=candidate_accepted,
                blocked_gate_edits=gate_accepted,
                reason="Final iteration or editing disabled for this iteration.",
                trace=trace,
                next_schema=schema,
            )

        if not operator_constraints_enabled:
            trace["notes"].append("Operator constraints disabled; schema edits cannot be committed.")
            return SchemaTransitionResult(
                decision="continue",
                next_action="continue_training",
                requested_next_action=requested_next_action,
                schema_changed=False,
                committed_edits=[],
                rejected_edits=final_plan,
                blocked_candidate_edits=candidate_accepted,
                blocked_gate_edits=gate_accepted,
                reason="Operator constraints disabled; no schema transition committed.",
                trace=trace,
                next_schema=schema,
            )

        if final_plan:
            try:
                next_schema = RewardEditOperatorApplier.apply(schema, final_plan)
                canonical_dict, canonical_report = SchemaCanonicalizer.canonicalize_schema(
                    next_schema.to_dict(),
                    primitive_interface=primitive_interface or {},
                    reward_blueprint={},
                )
                next_schema = RewardSchema.from_dict(canonical_dict)
                trace["schema_canonicalization_report"] = canonical_report
            except Exception as exc:
                trace["notes"].append(f"Schema apply failed: {exc}")
                return SchemaTransitionResult(
                    decision="reject",
                    next_action="continue_training",
                    requested_next_action=requested_next_action,
                    schema_changed=False,
                    committed_edits=[],
                    rejected_edits=final_plan,
                    blocked_candidate_edits=candidate_accepted,
                    blocked_gate_edits=gate_accepted,
                    reason=f"Committed edit application failed; continuing current schema. Error: {exc}",
                    trace=trace,
                    next_schema=schema,
                )

            return SchemaTransitionResult(
                decision="commit",
                next_action="apply_edit",
                requested_next_action=requested_next_action,
                schema_changed=True,
                committed_edits=final_plan,
                rejected_edits=[],
                blocked_candidate_edits=[],
                blocked_gate_edits=[],
                reason="Final edit plan committed to next schema.",
                trace=trace,
                next_schema=next_schema,
            )

        # At this point, candidate/gate may have accepted something earlier,
        # but final_plan is empty. That means a later stage, normally tool
        # audit/repair, blocked execution. Do not silently call it accepted.
        blocked_gate_edits = gate_accepted if gate_accepted else []
        blocked_candidate_edits = candidate_accepted if candidate_accepted else []

        if blocked_gate_edits:
            trace["notes"].append(
                "Gate accepted edits, but no final edit_plan survived later stages. "
                "Treating them as blocked_gate_edits, not committed_edits."
            )

        if requested_next_action == "early_stop":
            return SchemaTransitionResult(
                decision="stop",
                next_action="early_stop",
                requested_next_action=requested_next_action,
                schema_changed=False,
                committed_edits=[],
                rejected_edits=[],
                blocked_candidate_edits=blocked_candidate_edits,
                blocked_gate_edits=blocked_gate_edits,
                reason="No committed edit and requested_next_action=early_stop.",
                trace=trace,
                next_schema=schema,
            )

        if requested_next_action == "structural_search":
            reason = (
                "Structural search did not yield a committed edit in this iteration; "
                "continue current schema from latest checkpoint."
            )
            decision = "continue"
            resolved_next_action = "structural_search"
        else:
            reason = "No committed edit; continue current schema."
            decision = "continue"
            resolved_next_action = "continue_training"

        return SchemaTransitionResult(
            decision=decision,
            next_action=resolved_next_action,
            requested_next_action=requested_next_action,
            schema_changed=False,
            committed_edits=[],
            rejected_edits=final_plan,
            blocked_candidate_edits=blocked_candidate_edits,
            blocked_gate_edits=blocked_gate_edits,
            reason=reason,
            trace=trace,
            next_schema=schema,
        )

    @staticmethod
    def _get_list(obj: Any, attr: str) -> List[Dict[str, Any]]:
        if obj is None:
            return []
        if isinstance(obj, dict):
            value = obj.get(attr, [])
        else:
            value = getattr(obj, attr, [])
        if value is None:
            return []
        if isinstance(value, list):
            return copy.deepcopy(value)
        return []
