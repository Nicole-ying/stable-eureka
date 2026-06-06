from __future__ import annotations

import json
from typing import Any, Dict, List


def build_reflection_prompt(
    task_description: str,
    current_reward_schema: Dict[str, Any],
    diagnostic_report: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
    retrieved_lessons: List[Dict[str, Any]],
) -> str:
    return f"""
You are the ReflectionAgent of EG-RSA.
Your job is NOT to edit the reward. Your job is to analyze the current training evidence,
judge memory quality, and decide the next reward-search strategy for the editor.
Return one JSON object only.

Mission:
- Preserve the LLM's strategic reasoning as an explicit reflection report.
- Distinguish true reward hacking from task failure, detector false positives, and training insufficiency.
- Use memory as evidence, not as decoration.
- Decide whether the next edit should be a single edit, an atomic coupled rebalancing package,
  structural search, continue training, or early stop.

Hard constraints:
1. Do NOT use official/oracle reward for decisions.
2. Do NOT write Python code.
3. Do NOT invent metrics, events, variables, or operators.
4. Do NOT propose a final edit_plan here. Only provide strategy and constraints for the EditAgent.
5. If a failed lesson says an edit failed only because it was applied alone, preserve that nuance.

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

Return exactly this JSON format:
{{
  "reflection_summary": "short summary of what is happening",
  "failure_assessment": {{
    "observed_facts": [],
    "likely_true_failures": [],
    "likely_false_positives": [],
    "root_cause_hypotheses": [],
    "failure_kind": "reward_hack | task_failure | detector_false_positive | mixed | unclear",
    "confidence": 0.0
  }},
  "memory_assessment": {{
    "reusable_lessons": [],
    "failed_or_weak_lessons": [],
    "conflicting_lessons": [],
    "avoid_actions": [],
    "recommended_actions": [],
    "memory_confidence": 0.0
  }},
  "strategy": {{
    "recommended_next_action": "apply_edit | structural_search | continue_training | early_stop",
    "plan_type": "single_edit | coupled_rebalancing | structural_search | continue_training | early_stop",
    "atomicity": "atomic | separable",
    "why_atomic_or_separable": "If a decrease/increase rebalancing package must not be split, say so explicitly.",
    "max_reasonable_edits": 1,
    "editor_constraints": [],
    "must_preserve": [],
    "must_avoid": [],
    "expected_effect": "expected direction of task proxies and hack risk",
    "risk_analysis": "what could go wrong"
  }},
  "auditor_hints": {{
    "package_should_be_rejected_if": [],
    "package_should_be_accepted_if": [],
    "safety_notes": []
  }}
}}
""".strip()
