from __future__ import annotations

import json
from typing import Any, Dict, List


def _json_block(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_repair_prompt(
    task_description: str,
    current_reward_schema: Dict[str, Any],
    diagnostic_report: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
    retrieved_lessons: List[Dict[str, Any]],
    reflection_report: Dict[str, Any],
    failed_edit_response: Dict[str, Any],
    scale_audit_report: Dict[str, Any],
    behavior_risk_report: Dict[str, Any] | None = None,
) -> str:
    behavior_risk_report = behavior_risk_report or {}

    return f"""
You are the RepairAgent inside EG-RSA v1.

A previous reward edit plan was not directly executed because tool diagnostics
reported risk. The tools are observations, not final judges. You are the agent
that must understand the risk and produce a safer plan.

Important abstraction:
Do not reason only from environment-specific component names. Use semantic roles:

- terminal_success: sparse or event-style task completion / success reward
- dense_guidance: dense progress, navigation, shaping, or approach guidance
- stability_quality: quality, balance, smoothness, posture, safe-state shaping
- control_cost: energy, action magnitude, torque, effort, or control penalty
- safety_constraint: fall, crash, collision, or unsafe-state penalty

Tool reports:
- ScaleAudit checks whether reward magnitudes may dominate terminal incentives.
- BehaviorRiskAudit checks role-level behavior risk, e.g. terminal_success before
  stability, dense_guidance removed too early, control_cost overpressure,
  dense-role dominance transfer, or repeated role patterns from regression
  lessons.

Your tasks:
1. Identify whether the risk is scale risk, role-level behavior risk, or both.
2. Preserve the useful intent of the failed plan if possible.
3. Reduce aggressive role changes rather than blindly deleting all edits.
4. If terminal_success is increased while success/stability evidence is weak,
   prefer conservative increase and preserve dense_guidance or strengthen
   stability_quality.
5. If control_cost is increased while success is weak, use a small change or
   defer it until the task behavior is stable.
6. If BehaviorRiskAudit reports dense_role_dominance_transfer or
   dense_role_dominance_amplification, do not move exploitation from one dense
   role to another. Prefer smaller changes, preserve guidance, or choose
   continue_training.
7. If a retrieved lesson says a role/operator pattern regressed, explicitly
   explain why your repair will not repeat it.
8. If no safe repair exists, choose no_edit + continue_training.

Return valid JSON only.

Required JSON format:
{{
  "repair_analysis": {{
    "risk_source": "...",
    "role_level_interpretation": "...",
    "what_to_keep": "...",
    "what_to_modify": "...",
    "what_to_remove": "...",
    "why_repaired_plan_is_safer": "..."
  }},
  "diagnosis": "...",
  "reward_editor": {{
    "edit_decision": "edit | no_edit | need_more_evidence",
    "next_action": "apply_edit | continue_training | structural_search",
    "plan_type": "single_edit | coupled_rebalancing | structural_search | continue_training",
    "atomicity": "atomic | separable",
    "max_reasonable_edits": 1,
    "rationale": "..."
  }},
  "auditor_check": {{
    "approved": true,
    "issues": [],
    "final_action": "apply_edit | continue_training | structural_search"
  }},
  "edit_plan": []
}}

Task description:
{task_description}

Current reward schema:
{_json_block(current_reward_schema)}

Diagnostic report:
{_json_block(diagnostic_report)}

Reflection report:
{_json_block(reflection_report)}

Retrieved memory:
{_json_block(retrieved_memories)}

Retrieved lessons:
{_json_block(retrieved_lessons)}

Failed edit response:
{_json_block(failed_edit_response)}

ScaleAudit report:
{_json_block(scale_audit_report)}

BehaviorRiskAudit role-level report:
{_json_block(behavior_risk_report)}
""".strip()
