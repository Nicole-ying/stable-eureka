from __future__ import annotations

import json
from typing import Any, Dict, List

from eg_rsa.reward.operators import RewardEditOperatorApplier


def build_edit_prompt(
    task_description: str,
    current_reward_schema: Dict[str, Any],
    diagnostic_report: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
) -> str:
    """Build the constrained prompt for the EG-RSA edit agent.

    The model is explicitly forbidden from generating Python code.  It must only
    emit an edit-plan JSON object whose operators are executed by trusted code.
    """

    allowed_ops = RewardEditOperatorApplier.allowed_operator_descriptions()
    return f"""
You are the EG-RSA Reward Editing Agent.

You must NOT write Python code.
You must NOT invent new edit operators.
You must only return a valid JSON object.
Your job is to edit the current reward schema based on diagnostics and memory.

Task description:
{task_description}

Current reward schema:
{json.dumps(current_reward_schema, indent=2, ensure_ascii=False)}

Diagnostic report:
{json.dumps(diagnostic_report, indent=2, ensure_ascii=False)}

Retrieved memory cards:
{json.dumps(retrieved_memories, indent=2, ensure_ascii=False)}

Allowed edit operators:
{json.dumps(allowed_ops, indent=2, ensure_ascii=False)}

Return exactly this JSON format:
{{
  "diagnosis": "short explanation",
  "edit_plan": [
    {{"operator": "decrease_weight", "target": "component_name", "factor": 0.5}}
  ]
}}
""".strip()
