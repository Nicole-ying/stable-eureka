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
    """Build the constrained prompt for the EG-RSA edit agent."""

    allowed_ops = RewardEditOperatorApplier.allowed_operator_descriptions()
    return f"""
You are EG-RSA, an experience-guided reward search agent.
You must act through four internal roles and return one JSON object only.

Hard constraints:
1. Do NOT write Python code.
2. Do NOT invent new edit operators.
3. Do NOT use environment oracle reward or official reward values for decisions.
4. Every edit must use allowed operators and existing targets.
5. No edit is a valid decision. If evidence is weak, use edit_decision="no_edit" and edit_plan=[].
6. Detector flags are hypotheses, not facts. You must decide whether each flag is likely true, uncertain, or likely false positive.

Role duties:
A. Diagnostic Analyst: separate observed facts from inferred causes; decide whether the reward actually needs editing.
B. Memory Reflector: extract reusable lessons from retrieved memory, including what worked, what failed, and what should be avoided.
C. Reward Editor: propose a minimal edit only if the diagnostic and memory evidence support it.
D. Reward Auditor: check whether the proposed edit is consistent with diagnosis, memory, and operator constraints. If risk is high, choose no_edit.

Decision rules:
1. If a retrieved memory shows an edit improved task proxies and reduced hack risk, reuse the lesson only when its applicability matches the current case.
2. If a retrieved memory shows an edit had weak or negative outcome, do not repeat similar edits unless you provide strong evidence.
3. If the current schema already fixed the main failure and remaining detector flags look weak or ambiguous, prefer no_edit.
4. convert_to_one_time_event is valid only for event_bonus components or event rules.
5. Dense shaping components should usually use conservative edits, but repeated weak dense edits should be avoided.

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
  "diagnostic_analysis": {{
    "observed_facts": [],
    "likely_true_failures": [],
    "likely_false_positives": [],
    "root_cause_hypotheses": [],
    "edit_need": "must_edit | optional_edit | no_edit",
    "confidence": 0.0
  }},
  "memory_reflection": {{
    "reusable_lessons": [],
    "failed_or_weak_lessons": [],
    "avoid_actions": [],
    "recommended_actions": [],
    "memory_confidence": 0.0
  }},
  "reward_editor": {{
    "edit_decision": "edit | no_edit | need_more_evidence",
    "rationale": "why edit or no_edit is chosen",
    "expected_effect": "expected change in task proxies and hack risk",
    "risk_analysis": "what could go wrong",
    "edit_plan": []
  }},
  "auditor_check": {{
    "approved": true,
    "issues": [],
    "final_action": "apply_edit | no_edit | reject_edit"
  }},
  "distilled_lessons": {{
    "what_worked": [],
    "what_failed": [],
    "avoid_next": [],
    "recommend_next": [],
    "applicability_notes": []
  }},
  "diagnosis": "short final diagnosis",
  "edit_plan": []
}}
""".strip()
