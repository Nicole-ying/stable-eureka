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

Hard constraints:
1. You must NOT write Python code.
2. You must NOT invent new edit operators.
3. You must return exactly one valid JSON object and no Markdown.
4. Every edit must use one of the allowed operators.
5. Every target must exist in the current reward schema.
6. If you are uncertain, prefer a conservative single edit or return an empty edit_plan.

Editing principles:
1. First identify active failure_modes and the dominant reward component from diagnostics.
2. Then inspect retrieved memory cards. Prefer edits that improved similar failure modes before.
3. Avoid edits that previously worsened hack_score or task_score.
4. convert_to_one_time_event is only valid for event_bonus components or event rules.
5. Dense shaping components such as distance_penalty, velocity_penalty, angle_penalty, and action_penalty should usually use clip_component, decrease_weight, or disable_component, not one-time conversion.
6. If repeated_event_exploitation is active and the dominant component is event-like, consider convert_to_one_time_event.
7. If single_component_dominance is active, consider decrease_weight or clip_component on the dominant component.
8. Keep the edit_plan short: one edit is preferred unless diagnostics strongly support multiple edits.

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
  "failure_analysis": "what failure modes are active and why",
  "memory_usage": "how retrieved memory influenced the edit; say no relevant memory if empty",
  "operator_reasoning": "why this operator is valid for this target",
  "diagnosis": "short final diagnosis",
  "edit_plan": [
    {{"operator": "decrease_weight", "target": "component_name", "factor": 0.5}}
  ]
}}
""".strip()
