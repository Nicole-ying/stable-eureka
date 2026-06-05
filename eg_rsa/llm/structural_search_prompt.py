from __future__ import annotations

import json
from typing import Any, Dict, List

from eg_rsa.reward.operators import RewardEditOperatorApplier


def build_structural_search_prompt(
    task_description: str,
    current_reward_schema: Dict[str, Any],
    diagnostic_report: Dict[str, Any],
    retrieved_lessons: List[Dict[str, Any]],
    structural_context: Dict[str, Any],
) -> str:
    allowed_ops = RewardEditOperatorApplier.allowed_operator_descriptions()
    return f"""
You are the Structural Search sub-agent of EG-RSA.
Your job is to propose a generic structural reward edit when local edits are insufficient.
Return one JSON object only.

Hard constraints:
1. Do NOT use environment oracle reward or official reward values.
2. Do NOT write Python code.
3. Do NOT invent variables, events, or operators.
4. You may only use allowed operators and the available events/metrics listed in structural_context.
5. Prefer one minimal structural edit. The execution gate will keep at most one edit.
6. Do not add unprotected repeatable event bonuses. Use one_time=true and/or duration_steps when adding event rules.
7. This must be generic: no environment-specific hard-coded policy, only configured events/metrics and schema operators.

Structural search purpose:
- If a known exploit was removed but task guidance remains weak, add a non-exploitable positive or shaping signal using configured events/metrics.
- If current local edits failed, avoid repeating them.
- If no safe structural edit exists, return edit_decision="no_edit" and next_action="early_stop".

Required add_event_rule schema:
When using add_event_rule, event_rule MUST contain all fields below:
{{
  "operator": "add_event_rule",
  "event_rule": {{
    "name": "r_unique_rule_name",
    "type": "event_bonus",
    "weight": 20.0,
    "condition": {{"one_available_event_name": true, "duration_steps": 3}},
    "one_time": true,
    "enabled": true
  }}
}}
Do NOT return compact forms such as {{"event": "...", "weight": 100}}.

Task description:
{task_description}

Current reward schema:
{json.dumps(current_reward_schema, indent=2, ensure_ascii=False)}

Diagnostic report:
{json.dumps(diagnostic_report, indent=2, ensure_ascii=False)}

Retrieved lesson cards:
{json.dumps(retrieved_lessons, indent=2, ensure_ascii=False)}

Structural context:
{json.dumps(structural_context, indent=2, ensure_ascii=False)}

Allowed edit operators:
{json.dumps(allowed_ops, indent=2, ensure_ascii=False)}

Return exactly this JSON format:
{{
  "structural_analysis": {{
    "missing_signal_hypothesis": "what generic reward signal appears missing",
    "why_local_edit_is_insufficient": "why existing local operators are insufficient",
    "memory_constraints": [],
    "safety_constraints": []
  }},
  "reward_editor": {{
    "edit_decision": "edit | no_edit",
    "next_action": "apply_edit | early_stop",
    "rationale": "why this structural edit is safe and useful",
    "edit_plan": []
  }},
  "auditor_check": {{
    "approved": true,
    "issues": [],
    "final_action": "apply_edit | early_stop | reject_edit"
  }},
  "diagnosis": "short structural-search diagnosis",
  "edit_plan": []
}}
""".strip()
