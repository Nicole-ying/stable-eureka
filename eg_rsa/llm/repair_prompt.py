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

Tool reports:
- ScaleAudit checks whether reward magnitudes may dominate terminal incentives.
- BehaviorRiskAudit checks whether the edit may cause unstable contact,
  premature terminal pressure, loss of guidance, or repeated patterns from
  prior regression lessons.

Your tasks:
1. Identify whether the risk is scale risk, behavior risk, or both.
2. Preserve the useful intent of the failed plan if possible.
3. Reduce aggressive multipliers.
4. Avoid combining strong terminal pressure with strong energy penalty when
   success/stability evidence is weak.
5. Do not repeat edit patterns that retrieved outcome lessons mark as regression.
6. Prefer conservative local refinement over a large new dense penalty.
7. If no safe repair exists, choose no_edit + continue_training.

Return valid JSON only.

Required JSON format:
{{
  "repair_analysis": {{
    "risk_source": "...",
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

BehaviorRiskAudit report:
{_json_block(behavior_risk_report)}
""".strip()
