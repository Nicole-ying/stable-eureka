from __future__ import annotations

import json
from typing import Any, Dict, List

from eg_rsa.reward.operators import RewardEditOperatorApplier


def _pick_var(allowed_vars: List[str], candidates: List[str], fallback_index: int = 0) -> str:
    allowed = [str(x) for x in allowed_vars if str(x)]
    allowed_set = set(allowed)
    for name in candidates:
        if name in allowed_set:
            return name
    return allowed[min(fallback_index, len(allowed) - 1)] if allowed else "x"


def _pick_bool_like(allowed_vars: List[str], candidates: List[str], fallback: str) -> str:
    allowed = [str(x) for x in allowed_vars if str(x)]
    allowed_set = set(allowed)
    for name in candidates:
        if name in allowed_set:
            return name
    for name in allowed:
        low = name.lower()
        if any(k in low for k in ["leg", "contact", "touch", "ground", "support"]):
            return name
    return fallback


def _dynamic_vars(allowed_vars: List[str]) -> Dict[str, str]:
    x = _pick_var(
        allowed_vars,
        ["x_pos", "x", "horizontal_position", "horizontal_pos", "position_x", "target_x"],
        0,
    )
    y = _pick_var(
        allowed_vars,
        ["y_pos", "y", "vertical_position", "vertical_pos", "position_y", "height", "altitude"],
        1,
    )
    vx = _pick_var(
        allowed_vars,
        ["x_vel", "vx", "horizontal_velocity", "horizontal_speed", "velocity_x"],
        2,
    )
    vy = _pick_var(
        allowed_vars,
        ["y_vel", "vy", "vertical_velocity", "vertical_speed", "velocity_y"],
        3,
    )
    angle = _pick_var(
        allowed_vars,
        ["angle", "body_angle", "hull_angle", "tilt", "tilt_angle"],
        4,
    )
    left = _pick_bool_like(
        allowed_vars,
        ["left_leg", "left_contact", "left_leg_contact", "left_support", "left_touch"],
        x,
    )
    right = _pick_bool_like(
        allowed_vars,
        ["right_leg", "right_contact", "right_leg_contact", "right_support", "right_touch"],
        y,
    )
    main = _pick_var(allowed_vars, ["main_engine", "main_thrust", "thrust", "engine"], 0)
    side = _pick_var(allowed_vars, ["side_engine", "side_thrust", "steer", "torque"], 0)
    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "angle": angle,
        "left": left,
        "right": right,
        "main": main,
        "side": side,
    }


def _ast_grammar(allowed_vars: List[str]) -> Dict[str, Any]:
    v = _dynamic_vars(allowed_vars)
    return {
        "leaf": [{"var": v["x"]}, {"const": 0.5}, {"bool": True}],
        "numeric": [
            {"op": "add", "args": [{"var": v["x"]}, {"var": v["y"]}]},
            {"op": "sub", "left": {"const": 1.0}, "right": {"var": v["x"]}},
            {"op": "mul", "args": [{"const": 0.5}, {"op": "abs", "arg": {"var": v["vx"]}}]},
            {"op": "neg", "arg": {"var": v["main"]}},
            {"op": "abs", "arg": {"var": v["angle"]}},
            {"op": "min", "args": [{"var": v["x"]}, {"const": 1.0}]},
            {"op": "max", "args": [{"var": v["x"]}, {"const": 0.0}]},
            {"op": "clip", "args": [{"var": v["x"]}, {"const": -1.0}, {"const": 1.0}]},
        ],
        "boolean": [
            {"op": "and", "args": [
                {"op": "gt", "left": {"var": v["left"]}, "right": {"const": 0.5}},
                {"op": "gt", "left": {"var": v["right"]}, "right": {"const": 0.5}},
            ]},
            {"op": "lt", "left": {"op": "abs", "arg": {"var": v["vy"]}}, "right": {"const": 0.4}},
            {"op": "lt", "left": {"op": "abs", "arg": {"var": v["angle"]}}, "right": {"const": 0.3}},
            {"op": "or", "args": [
                {"op": "gt", "left": {"var": v["left"]}, "right": {"const": 0.5}},
                {"op": "gt", "left": {"var": v["right"]}, "right": {"const": 0.5}},
            ]},
        ],
        "note": "Example variables are dynamically selected from Allowed variables. Do not use variable names outside Allowed variables.",
    }


def _example_replace_formula(allowed_vars: List[str]) -> Dict[str, Any]:
    v = _dynamic_vars(allowed_vars)
    return {
        "operator": "replace_formula",
        "target": "r_progress_guidance",
        "formula_ast": {
            "op": "sub",
            "left": {"const": 1.0},
            "right": {
                "op": "min",
                "args": [
                    {"op": "abs", "arg": {"var": v["x"]}},
                    {"const": 1.0},
                ],
            },
        },
    }


def _example_replace_condition(allowed_vars: List[str]) -> Dict[str, Any]:
    v = _dynamic_vars(allowed_vars)
    return {
        "operator": "replace_condition",
        "target": "r_primitive_terminal_success",
        "condition_ast": {
            "op": "and",
            "args": [
                {"op": "gt", "left": {"var": v["left"]}, "right": {"const": 0.5}},
                {"op": "gt", "left": {"var": v["right"]}, "right": {"const": 0.5}},
                {"op": "lt", "left": {"op": "abs", "arg": {"var": v["vy"]}}, "right": {"const": 0.4}},
                {"op": "lt", "left": {"op": "abs", "arg": {"var": v["angle"]}}, "right": {"const": 0.3}},
            ],
        },
    }


def _example_add_formula_component(allowed_vars: List[str]) -> Dict[str, Any]:
    v = _dynamic_vars(allowed_vars)
    formula_ast = {
        "op": "sub",
        "left": {"const": 1.0},
        "right": {
            "op": "min",
            "args": [
                {"op": "abs", "arg": {"var": v["x"]}},
                {"const": 1.0},
            ],
        },
    }
    return {
        "operator": "add_formula_component",
        "component": {
            "name": "r_new_progress_ast",
            "type": "formula_component",
            "weight": 0.5,
            "formula_ast": formula_ast,
            "params": {"formula_ast": formula_ast},
            "clip": [0.0, 1.0],
            "enabled": True,
            "semantic_role": "dense_guidance",
            "reward_timing": "dense",
            "behavior_channel": "progress",
        },
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

    grammar = _ast_grammar(allowed_vars)
    example_replace_formula = _example_replace_formula(allowed_vars)
    example_replace_condition = _example_replace_condition(allowed_vars)
    example_add_formula = _example_add_formula_component(allowed_vars)

    return f"""
You are the Reward EditAgent of EG-RSA-V2 AST-IR.

Return one JSON object only.

Hard AST constraints:
1. Do NOT write Python code.
2. Do NOT output string formula, string condition, expression, or formula fields.
3. Formula edits must use formula_ast.
4. Condition edits must use condition_ast or condition.expr_ast.
5. AST variables must come from Allowed variables.
6. Do not copy variable names from generic examples unless they are listed in Allowed variables.
7. Prefer minimal edits. If no reliable edit exists, choose continue_training or structural_search.

Allowed variables:
{json.dumps(allowed_vars, ensure_ascii=False)}

AST grammar:
{json.dumps(grammar, indent=2, ensure_ascii=False)}

Allowed edit operators:
{json.dumps(allowed_ops, indent=2, ensure_ascii=False)}

Examples using the current allowed-variable contract:
- replace_formula:
{json.dumps(example_replace_formula, indent=2, ensure_ascii=False)}

- replace_condition:
{json.dumps(example_replace_condition, indent=2, ensure_ascii=False)}

- add_formula_component:
{json.dumps(example_add_formula, indent=2, ensure_ascii=False)}

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
    "plan_type": "single_edit | coupled_rebalancing | structural_search | continue_training",
    "atomicity": "atomic | separable",
    "max_reasonable_edits": 1,
    "rationale": "..."
  }},
  "auditor_check": {{
    "approved": true,
    "issues": [],
    "final_action": "apply_edit | continue_training | structural_search | early_stop"
  }},
  "distilled_lessons": {{
    "new_lesson_candidates": []
  }},
  "diagnosis": "...",
  "plan_type": "single_edit | coupled_rebalancing | structural_search | continue_training",
  "atomicity": "atomic | separable",
  "max_reasonable_edits": 1,
  "edit_plan": []
}}
""".strip()
