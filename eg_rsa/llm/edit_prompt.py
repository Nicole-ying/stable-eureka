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
    reflection_report: Dict[str, Any] | None = None,
) -> str:
    """Build the formula-native constrained prompt for the EG-RSA edit agent."""

    allowed_ops = RewardEditOperatorApplier.allowed_operator_descriptions()

    return f"""
You are the Reward EditAgent of EG-RSA-V2.
A separate ReflectionAgent has already analyzed diagnostics and memory.
Your job is to turn that reflection strategy into a concrete executable reward edit plan.

Return one JSON object only.

Mission constraints:
1. Improve reward search through memory-guided and reflection-guided editing.
2. Do NOT use environment oracle reward or official reward values for decisions.
3. Do NOT write Python code.
4. Do NOT invent new edit operators.
5. Prefer formula-native edits for V2 schemas:
   - replace_formula
   - replace_condition
   - add_formula_component
   - add_conditional_formula_component
   - add_action_penalty
   - add_event_predicate
6. Use metric_value / metric_delta only when the current structural context explicitly exposes a primitive-generated metric and a formula-native edit is not suitable.
7. Formula and condition edits must use only primitive variables already present in the schema/interface.
8. Do not merely tune weights if the root cause is missing progress structure or wrong formula/condition structure.
9. If a component is useful in intent but unsafe in formula, prefer replace_formula or replace_condition over disable_component.
10. If no reliable edit exists, choose continue_training or structural_search.

Formula-native design principles:
1. Dense reward should be progress-aligned, not merely passive state maintenance.
2. Stability/control components should support task progress rather than replace it.
3. Terminal success should be sparse, one-time, and based on primitive terminal evidence.
4. Action penalty must not become positive reward for any action direction.
5. A coupled edit package may combine weight change + formula/condition repair when both are required.

Allowed next_action values:
- apply_edit: use edit_plan to update the schema.
- structural_search: the issue needs a new formula-native structural signal.
- continue_training: the reward is plausible and insufficient training is the main hypothesis.
- early_stop: no reliable edit or search direction exists.

Task description:
{task_description}

Current reward schema:
{json.dumps(current_reward_schema, indent=2, ensure_ascii=False)}

Diagnostic report:
{json.dumps(diagnostic_report, indent=2, ensure_ascii=False)}

Reflection report:
{json.dumps(reflection_report or {}, indent=2, ensure_ascii=False)}

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
    "lesson_assessments": [],
    "reusable_lessons": [],
    "failed_or_weak_lessons": [],
    "avoid_actions": [],
    "recommended_actions": [],
    "memory_confidence": 0.0
  }},
  "reward_editor": {{
    "edit_decision": "edit | no_edit | need_more_evidence",
    "next_action": "apply_edit | structural_search | continue_training | early_stop",
    "plan_type": "single_edit | coupled_rebalancing | structural_search | continue_training | early_stop",
    "atomicity": "atomic | separable",
    "max_reasonable_edits": 1,
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
  "edit_plan": [],
  "plan_type": "single_edit | coupled_rebalancing | structural_search | continue_training | early_stop",
  "atomicity": "atomic | separable",
  "max_reasonable_edits": 1
}}
""".strip()
