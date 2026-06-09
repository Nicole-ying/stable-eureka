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
) -> str:
    """Prompt for repairing a risky edit_plan after tool feedback.

    ScaleAudit is a tool observation, not a final judge. The LLM should use the
    report to understand what made the proposed edit unsafe and then produce a
    safer plan that preserves the original intent when possible.
    """

    return f"""
You are the RepairAgent inside EG-RSA v1.

Your previous reward edit plan was NOT executed because ScaleAudit reported
that it may dominate terminal incentives or otherwise destabilize the reward
scale.

Your task:
1. Understand why the previous edit was risky.
2. Preserve the useful intent of the plan if possible.
3. Repair the edit plan by changing scale, removing unsafe new dense terms, or
   replacing structural edits with conservative local edits.
4. Do not simply repeat the failed plan.
5. If no safe repair exists, choose no_edit + continue_training.

Important principles:
- ScaleAudit is tool evidence for you to reason over.
- You are the decision maker; the tool is not the agent.
- Prefer smaller local edits over large new dense penalties.
- If adding a new dense component, its expected episode-level contribution must
  be much smaller than terminal success rewards.
- Keep atomic package coherence. If the original package was atomic, explain how
  the repaired package remains coherent.
- Official/oracle reward must not be used as an edit target.

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
""".strip()
