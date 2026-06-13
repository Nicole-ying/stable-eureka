from __future__ import annotations

import json
from typing import Any, Dict, List

from eg_rsa.reward.operators import RewardEditOperatorApplier


def _ast_grammar() -> Dict[str, Any]:
    return {
        "leaf": [{"var": "x"}, {"const": 0.5}, {"bool": True}],
        "numeric": [
            {"op": "add", "args": [{"var": "x"}, {"var": "y"}]},
            {"op": "sub", "left": {"const": 1.0}, "right": {"var": "x"}},
            {"op": "mul", "args": [{"const": 0.5}, {"op": "abs", "arg": {"var": "vx"}}]},
            {"op": "neg", "arg": {"var": "main_engine"}},
            {"op": "abs", "arg": {"var": "angle"}},
            {"op": "min", "args": [{"var": "x"}, {"const": 1.0}]},
            {"op": "max", "args": [{"var": "x"}, {"const": 0.0}]},
            {"op": "clip", "args": [{"var": "x"}, {"const": -1.0}, {"const": 1.0}]},
        ],
        "boolean": [
            {"op": "and", "args": [{"var": "left_contact"}, {"var": "right_contact"}]},
            {"op": "or", "args": [{"var": "left_contact"}, {"var": "right_contact"}]},
            {"op": "not", "arg": {"var": "left_contact"}},
            {"op": "lt", "left": {"op": "abs", "arg": {"var": "vy"}}, "right": {"const": 0.4}},
            {"op": "gt", "left": {"op": "abs", "arg": {"var": "angle"}}, "right": {"const": 0.6}},
        ],
    }


def build_edit_prompt(
    task_description: str,
    current_reward_schema: Dict[str, Any],
    diagnostic_report: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
    retrieved_lessons: List[Dict[str, Any]] = None,
    reflection_report: Dict[str, Any] = None,
) -> str:
    allowed_ops = RewardEditOperatorApplier.allowed_operator_descriptions()
    allowed_vars = (
        current_reward_schema.get("metadata", {}).get("allowed_formula_variables", [])
        if isinstance(current_reward_schema, dict)
        else []
    )

    return f"""
You are the Reward EditAgent of EG-RSA-V2 AST-IR.

Return one JSON object only.

Hard AST constraints:
1. Do NOT write Python code.
2. Do NOT output string formula, string condition, expression, or formula fields.
3. Formula edits must use formula_ast.
4. Condition edits must use condition_ast or condition.expr_ast.
5. AST variables must come from allowed variables.
6. Prefer minimal edits. If no reliable edit exists, choose continue_training or structural_search.

Allowed variables:
{json.dumps(allowed_vars, ensure_ascii=False)}

AST grammar:
{json.dumps(_ast_grammar(), indent=2, ensure_ascii=False)}

Allowed edit operators:
{json.dumps(allowed_ops, indent=2, ensure_ascii=False)}

Examples:
- replace_formula:
{{
  "operator": "replace_formula",
  "target": "r_progress_guidance",
  "formula_ast": {{"op": "sub", "left": {{"const": 1.0}}, "right": {{"op": "min", "args": [{{"op": "abs", "arg": {{"var": "x"}}}}, {{"const": 1.0}}]}}}}
}}

- replace_condition:
{{
  "operator": "replace_condition",
  "target": "r_primitive_terminal_success",
  "condition_ast": {{"op": "and", "args": [{{"var": "left_contact"}}, {{"var": "right_contact"}}]}}
}}

- add_formula_component:
{{
  "operator": "add_formula_component",
  "component": {{
    "name": "r_new_progress_ast",
    "type": "formula_component",
    "weight": 0.5,
    "formula_ast": {{"op": "sub", "left": {{"const": 1.0}}, "right": {{"op": "min", "args": [{{"op": "abs", "arg": {{"var": "x"}}}}, {{"const": 1.0}}]}}}},
    "params": {{"formula_ast": {{"op": "sub", "left": {{"const": 1.0}}, "right": {{"op": "min", "args": [{{"op": "abs", "arg": {{"var": "x"}}}}, {{"const": 1.0}}]}}}}}},
    "clip": [0.0, 1.0],
    "enabled": true,
    "semantic_role": "dense_guidance",
    "reward_timing": "dense",
    "behavior_channel": "progress"
  }}
}}

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
