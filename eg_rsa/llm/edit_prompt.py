from __future__ import annotations

import json
from typing import Any, Dict, List

from eg_rsa.reward.operators import RewardEditOperatorApplier


def build_edit_prompt(
    task_description: str,
    current_reward_schema: Dict[str, Any],
    diagnostic_report: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
    retrieved_lessons: List[Dict[str, Any]] | None = None,
) -> str:
    """Build the constrained prompt for the EG-RSA edit agent."""

    allowed_ops = RewardEditOperatorApplier.allowed_operator_descriptions()
    return f"""
You are EG-RSA, an experience-guided reward search agent.
You must act through four internal roles and return one JSON object only.

Mission constraints:
1. Improve reward search through diagnosis, memory, editing, auditing, and self-evolution.
2. Do NOT use environment oracle reward or official reward values for decisions.
3. Do NOT use environment-specific hard-coded rules; use only task description, diagnostics, schema, allowed operators, and retrieved memory.
4. Do NOT write Python code.
5. Do NOT invent new edit operators.
6. Every edit must use allowed operators and existing targets.

Critical distinction:
A detector flag is only a hypothesis. You must distinguish:
- reward_hack: the behavior actually increases the learned reward;
- task_failure: the behavior is bad but does not exploit the learned reward;
- detector_false_positive: the detector fires but the current reward structure no longer supports exploitation.

Memory usage rules:
1. Raw memory cards are factual records of past edit trials.
2. Distilled lesson cards are reusable experience extracted from raw memory.
3. Do not merely mention retrieved lessons. Evaluate their quality and applicability.
4. Classify each relevant lesson as one of: reusable_now, already_applied, not_applicable, weak_or_failed, conflicting.
5. Reuse successful lessons only when their applicability matches the current case.
6. Avoid repeating failed or weak lessons unless you provide strong evidence.
7. If lessons conflict, state the conflict and choose the safer action.

Role duties:
A. Diagnostic Analyst: separate observed facts from inferred causes; decide whether the reward actually needs editing.
B. Memory Reflector: assess lesson quality, applicability, conflicts, and whether a lesson is already absorbed by the current schema.
C. Reward Editor: choose one next action. Do not waste another iteration by repeating the same schema without a reason.
D. Reward Auditor: check whether the action is consistent with diagnosis, memory, and operator constraints. If risk is high, reject or choose a safer action.

Allowed next_action values:
- apply_edit: use edit_plan to update the schema.
- structural_search: the issue is not a current hack but poor task guidance; propose a generic structural reward-search direction using allowed operators if possible.
- continue_training: the reward is plausible and insufficient training is the main hypothesis; use sparingly.
- early_stop: no reliable edit or search direction exists; stop this reward-search run.

Decision rules:
1. If edit_decision="no_edit", next_action must explain what happens next; do not leave it as a wasted repeated training run.
2. If task proxies remain poor but reward-hack evidence is weak, prefer structural_search over plain no_edit.
3. If the current schema already fixed the main failure and remaining detector flags are likely false positives, do not edit that same failure again.
4. If a retrieved lesson is already applied by the current schema, classify it as already_applied rather than reusable_now.
5. convert_to_one_time_event is valid only for event_bonus components or event rules.
6. add_duration_condition is valid only if the target is an event rule and sustained satisfaction is meaningful.
7. Dense shaping components should use conservative edits, but repeated weak dense edits should be avoided.

Task description:
{task_description}

Current reward schema:
{json.dumps(current_reward_schema, indent=2, ensure_ascii=False)}

Diagnostic report:
{json.dumps(diagnostic_report, indent=2, ensure_ascii=False)}

Raw memory cards:
{json.dumps(retrieved_memories, indent=2, ensure_ascii=False)}

Distilled lesson cards:
{json.dumps(retrieved_lessons or [], indent=2, ensure_ascii=False)}

Allowed edit operators:
{json.dumps(allowed_ops, indent=2, ensure_ascii=False)}

Return exactly this JSON format:
{{
  "diagnostic_analysis": {{
    "observed_facts": [],
    "likely_true_failures": [],
    "likely_false_positives": [],
    "root_cause_hypotheses": [],
    "failure_kind": "reward_hack | task_failure | detector_false_positive | mixed | unclear",
    "edit_need": "must_edit | optional_edit | no_edit",
    "confidence": 0.0
  }},
  "memory_reflection": {{
    "lesson_assessments": [
      {{
        "lesson_id": "lesson id if available",
        "quality": "strong | moderate | weak",
        "status": "reusable_now | already_applied | not_applicable | weak_or_failed | conflicting",
        "reason": "why this lesson should or should not affect the current decision"
      }}
    ],
    "reusable_lessons": [],
    "failed_or_weak_lessons": [],
    "avoid_actions": [],
    "recommended_actions": [],
    "memory_confidence": 0.0
  }},
  "reward_editor": {{
    "edit_decision": "edit | no_edit | need_more_evidence",
    "next_action": "apply_edit | structural_search | continue_training | early_stop",
    "rationale": "why this action is chosen",
    "expected_effect": "expected change in task proxies and hack risk",
    "risk_analysis": "what could go wrong",
    "edit_plan": []
  }},
  "auditor_check": {{
    "approved": true,
    "issues": [],
    "final_action": "apply_edit | structural_search | continue_training | early_stop | reject_edit"
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
