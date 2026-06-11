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
    allowed_vars = structural_context.get("allowed_formula_variables", [])
    allowed_funcs = structural_context.get("allowed_formula_functions", [])

    return f"""
You are the Structural Search sub-agent of EG-RSA-V2.

Your job is to propose one minimal formula-native structural reward edit when local edits are insufficient.
Return one JSON object only.

Primitive-only constraints:
1. Do NOT use environment oracle reward or official reward values.
2. Do NOT write Python code.
3. Do NOT invent hidden task metrics, hidden event predicates, hidden variables, or hidden rewards.
4. Prefer primitive formula edits over metric-based edits.
5. Formula expressions may use only these variables: {json.dumps(allowed_vars, ensure_ascii=False)}
6. Formula expressions may call only these functions: {json.dumps(allowed_funcs, ensure_ascii=False)}
7. If no safe structural edit exists, return edit_decision="no_edit" and next_action="continue_training".

Preferred structural edits:
1. add_formula_component
2. add_conditional_formula_component
3. add_action_penalty
4. add_event_predicate

Avoid metric_value / metric_delta unless there is no formula-native alternative.

Required add_formula_component schema:
{{
  "operator": "add_formula_component",
  "component": {{
    "name": "r_unique_formula_component",
    "type": "formula_component",
    "weight": 1.0,
    "formula": "1.0 - min(abs(x), 1.0)",
    "params": {{"formula": "1.0 - min(abs(x), 1.0)"}},
    "clip": [0.0, 1.0],
    "enabled": true,
    "semantic_role": "dense_guidance",
    "reward_timing": "dense",
    "behavior_channel": "progress"
  }}
}}

Required add_conditional_formula_component schema:
{{
  "operator": "add_conditional_formula_component",
  "component": {{
    "name": "r_unique_conditional_component",
    "type": "conditional_formula_component",
    "weight": 1.0,
    "condition": "some primitive condition",
    "formula": "some primitive formula",
    "params": {{
      "condition": "some primitive condition",
      "formula": "some primitive formula"
    }},
    "clip": [0.0, 1.0],
    "enabled": true,
    "semantic_role": "dense_guidance",
    "reward_timing": "dense",
    "behavior_channel": "progress"
  }}
}}

Required add_event_predicate schema:
{{
  "operator": "add_event_predicate",
  "event_rule": {{
    "name": "r_unique_event_predicate",
    "type": "event_predicate",
    "weight": 20.0,
    "condition": {{
      "expression": "primitive boolean expression",
      "duration_steps": 1
    }},
    "one_time": true,
    "enabled": true,
    "semantic_role": "terminal_success",
    "reward_timing": "sparse_event",
    "behavior_channel": "completion"
  }}
}}

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
    "missing_signal_hypothesis": "what primitive-variable reward signal appears missing",
    "why_local_edit_is_insufficient": "why existing local operators are insufficient",
    "memory_constraints": [],
    "safety_constraints": []
  }},
  "reward_editor": {{
    "edit_decision": "edit | no_edit",
    "next_action": "apply_edit | continue_training | early_stop",
    "rationale": "why this structural edit is safe and useful",
    "edit_plan": []
  }},
  "auditor_check": {{
    "approved": true,
    "issues": [],
    "final_action": "apply_edit | continue_training | early_stop | reject_edit"
  }},
  "diagnosis": "short structural-search diagnosis",
  "edit_plan": []
}}
""".strip()
