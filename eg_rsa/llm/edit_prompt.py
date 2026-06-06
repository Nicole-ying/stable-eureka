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
    """Build the constrained prompt for the EG-RSA edit agent."""

    allowed_ops = RewardEditOperatorApplier.allowed_operator_descriptions()
    return f"""
You are the Reward EditAgent of EG-RSA.
A separate ReflectionAgent has already analyzed the diagnostics and memory. Your job is to turn
that reflection strategy into a concrete executable reward edit plan.
Return one JSON object only.

Mission constraints:
1. Improve reward search through memory-guided and reflection-guided editing.
2. Do NOT use environment oracle reward or official reward values for decisions.
3. Do NOT use environment-specific hard-coded rules; use only task description, diagnostics, schema, allowed operators, retrieved memory, and reflection report.
4. Do NOT write Python code.
5. Do NOT invent new edit operators.
6. Every edit must use allowed operators and existing targets, except add_component/add_event_rule may create new schema items using configured metrics/events.

Reflection alignment rules:
1. Follow the ReflectionAgent strategy unless you explicitly explain why it is unsafe.
2. If reflection says plan_type="coupled_rebalancing" and atomicity="atomic", output a coherent package and mark it atomic.
3. Do NOT let a coupled package degrade into only the negative/decrease edit. If you decrease one dense shaping component to reduce dominance, pair it with the intended positive/completion/process support edits.
4. If failed lessons say "solo decrease_weight failed", avoid repeating solo decrease_weight.
5. If you cannot form a safe package, choose structural_search or continue_training rather than a weak single edit.

Critical distinction:
A detector flag is only a hypothesis. You must distinguish:
- reward_hack: the behavior actually increases the learned reward;
- task_failure: the behavior is bad but does not exploit the learned reward;
- detector_false_positive: the detector fires but the current reward structure no longer supports exploitation.

Memory usage rules:
1. Raw memory cards are factual records of past edit trials.
2. Distilled lesson cards are reusable experience extracted from raw memory.
3. Do not merely mention retrieved lessons. Evaluate their quality and applicability.
4. Reuse successful lessons only when their applicability matches the current case.
5. Avoid repeating failed or weak lessons unless you provide strong evidence and a materially different package.

Allowed next_action values:
- apply_edit: use edit_plan to update the schema.
- structural_search: the issue is not a current hack but poor task guidance; propose a generic structural reward-search direction using allowed operators if possible.
- continue_training: the reward is plausible and insufficient training is the main hypothesis; use sparingly.
- early_stop: no reliable edit or search direction exists; stop this reward-search run.

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
    "lesson_assessments": [
      {{
        "lesson_id": "lesson id if available",
        "quality": "strong | moderate | weak | failed",
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
