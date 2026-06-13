#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] Check repo layout..."
test -f train_eg_rsa.py
test -d eg_rsa
test -f eg_rsa/schema_sources/llm_bootstrap.py
test -f eg_rsa/llm/edit_prompt.py
test -f eg_rsa/reward/edit_plan_validator.py
test -f eg_rsa/runner.py

echo "[2/7] Backup files..."
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR=".eg_rsa_edit_contract_patch_backup_${TS}"
mkdir -p "${BACKUP_DIR}"

cp eg_rsa/schema_sources/llm_bootstrap.py "${BACKUP_DIR}/llm_bootstrap.py"
cp eg_rsa/llm/edit_prompt.py "${BACKUP_DIR}/edit_prompt.py"
cp eg_rsa/reward/edit_plan_validator.py "${BACKUP_DIR}/edit_plan_validator.py"
cp eg_rsa/runner.py "${BACKUP_DIR}/runner.py"

if [ -f run_landing_10iter_6seeds_2M.sh ]; then
  cp run_landing_10iter_6seeds_2M.sh "${BACKUP_DIR}/run_landing_10iter_6seeds_2M.sh"
fi

echo "[3/7] Patch llm_bootstrap.py: generate richer runtime diagnostics..."
python - <<'PY'
from pathlib import Path
import re

p = Path("eg_rsa/schema_sources/llm_bootstrap.py")
text = p.read_text(encoding="utf-8")

pattern = re.compile(
    r"    @staticmethod\n"
    r"    def _build_runtime_spec_from_primitive_interface\(primitive_interface: Dict\[str, Any\]\) -> Dict\[str, Any\]:\n"
    r".*?"
    r"    @staticmethod\n"
    r"    def _write_json",
    re.S,
)

replacement = r'''    @staticmethod
    def _build_runtime_spec_from_primitive_interface(primitive_interface: Dict[str, Any]) -> Dict[str, Any]:
        """Build executable diagnostics from the verified primitive interface.

        This function must use the same canonical variable names as the reward
        schema. It intentionally avoids official/oracle reward. The generated
        events/metrics are semantic rollout probes for EG-RSA reflection/editing.

        Important:
          - bool leg/contact variables should become contact evidence events.
          - landing-like tasks need stable_landing_condition and success proxies.
          - these diagnostics are not used as reward unless a schema explicitly
            references them; they are mainly for analysis/edit decisions.
        """
        observation_mapping = primitive_interface.get("observation_mapping")
        if not isinstance(observation_mapping, dict) or not observation_mapping:
            observation_mapping = {}
            for idx, item in enumerate(primitive_interface.get("observation_variables", []) or []):
                if isinstance(item, dict) and item.get("name"):
                    observation_mapping[str(item["name"])] = idx
        else:
            observation_mapping = {str(k): int(v) for k, v in observation_mapping.items()}

        observation_variables = primitive_interface.get("observation_variables", []) or []
        action_variables = primitive_interface.get("action_variables", []) or []

        bool_vars = []
        numeric_vars = []
        descriptions: Dict[str, str] = {}

        for item in observation_variables:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            desc = str(item.get("description", "") or "")
            descriptions[name] = desc
            typ = str(item.get("type", "")).lower()
            if typ in {"bool", "boolean"}:
                bool_vars.append(name)
            else:
                numeric_vars.append(name)

        action_names = [
            str(item["name"])
            for item in action_variables
            if isinstance(item, dict) and item.get("name")
        ]

        def has_var(name: str) -> bool:
            return name in observation_mapping

        def first_existing(candidates: list[str]) -> str | None:
            for name in candidates:
                if has_var(name):
                    return name
            return None

        def name_or_desc_contains(name: str, keywords: list[str]) -> bool:
            hay = f"{name} {descriptions.get(name, '')}".lower()
            return any(k.lower() in hay for k in keywords)

        # Canonical semantic slots. These names match the current source-aware
        # landing interface but also keep compatibility with older aliases.
        x_var = first_existing([
            "x_pos", "x", "horizontal_position", "horizontal_pos", "position_x",
            "target_x", "relative_x",
        ])
        y_var = first_existing([
            "y_pos", "y", "vertical_position", "vertical_pos", "position_y",
            "height", "altitude",
        ])
        x_vel_var = first_existing([
            "x_vel", "vx", "horizontal_velocity", "horizontal_speed", "velocity_x",
        ])
        y_vel_var = first_existing([
            "y_vel", "vy", "vertical_velocity", "vertical_speed", "velocity_y",
        ])
        angle_var = first_existing([
            "angle", "body_angle", "hull_angle", "tilt", "tilt_angle",
        ])
        ang_vel_var = first_existing([
            "ang_vel", "angular_velocity", "angular_vel",
            "body_angular_velocity", "hull_angular_velocity",
        ])

        left_contact_var = None
        right_contact_var = None

        for name in bool_vars:
            if left_contact_var is None and name_or_desc_contains(name, ["left", "leg", "contact", "touch", "ground", "support"]):
                if name_or_desc_contains(name, ["left"]):
                    left_contact_var = name
            if right_contact_var is None and name_or_desc_contains(name, ["right", "leg", "contact", "touch", "ground", "support"]):
                if name_or_desc_contains(name, ["right"]):
                    right_contact_var = name

        # Fallback when names are simply left_leg/right_leg or similar.
        if left_contact_var is None:
            left_contact_var = first_existing(["left_leg", "left_contact", "left_leg_contact", "left_support", "left_touch"])
        if right_contact_var is None:
            right_contact_var = first_existing(["right_leg", "right_contact", "right_leg_contact", "right_support", "right_touch"])

        contact_like = []
        for name in bool_vars:
            if name_or_desc_contains(name, ["contact", "touch", "ground", "support", "leg"]):
                contact_like.append(name)

        # Preserve left/right order when known.
        ordered_contact_like = []
        for name in [left_contact_var, right_contact_var]:
            if name and name not in ordered_contact_like:
                ordered_contact_like.append(name)
        for name in contact_like:
            if name not in ordered_contact_like:
                ordered_contact_like.append(name)
        contact_like = ordered_contact_like

        events: Dict[str, Dict[str, Any]] = {}

        # Expose bool observation variables as threshold events.
        for name in bool_vars:
            events[name] = {"type": "threshold_gt", "var": name, "threshold": 0.5}

        if left_contact_var:
            events["left_leg_contact"] = {"type": "threshold_gt", "var": left_contact_var, "threshold": 0.5}
        if right_contact_var:
            events["right_leg_contact"] = {"type": "threshold_gt", "var": right_contact_var, "threshold": 0.5}

        if contact_like:
            events["any_contact"] = {
                "type": "any",
                "events": [
                    "left_leg_contact" if name == left_contact_var else
                    "right_leg_contact" if name == right_contact_var else
                    name
                    for name in contact_like[:4]
                ],
            }

        if left_contact_var and right_contact_var:
            events["both_contact"] = {
                "type": "all",
                "events": ["left_leg_contact", "right_leg_contact"],
            }
        elif len(contact_like) >= 2:
            events["both_contact"] = {"type": "all", "events": contact_like[:2]}

        # Landing semantic events. Thresholds are conservative and only used as
        # diagnostic probes, not as official reward.
        if x_var:
            events["near_center"] = {"type": "threshold_abs", "var": x_var, "threshold": 0.25}
        if x_vel_var:
            events["low_horizontal_speed"] = {"type": "threshold_abs", "var": x_vel_var, "threshold": 0.5}
        if y_vel_var:
            events["low_vertical_speed"] = {"type": "threshold_abs", "var": y_vel_var, "threshold": 0.5}
        if angle_var:
            events["upright"] = {"type": "threshold_abs", "var": angle_var, "threshold": 0.25}
        if ang_vel_var:
            events["low_angular_speed"] = {"type": "threshold_abs", "var": ang_vel_var, "threshold": 0.5}

        safe_contact_children = [name for name in ["both_contact", "low_vertical_speed", "upright"] if name in events]
        if len(safe_contact_children) >= 2:
            events["safe_contact"] = {"type": "all", "events": safe_contact_children}

        stable_children = [
            name for name in [
                "both_contact",
                "near_center",
                "low_horizontal_speed",
                "low_vertical_speed",
                "upright",
                "low_angular_speed",
            ]
            if name in events
        ]
        if len(stable_children) >= 3:
            events["stable_landing_condition"] = {"type": "all", "events": stable_children}

        if action_names:
            events["action_nonzero"] = {"type": "action_nonzero"}

        task_metrics: Dict[str, Dict[str, Any]] = {}

        if x_var:
            task_metrics["approach_region_score"] = {
                "type": "target_region",
                "inputs": [x_var],
                "center": [0.0],
                "tolerance": [0.25],
            }

        if x_var and y_var:
            task_metrics["position_centering"] = {
                "type": "distance_to_target",
                "inputs": [x_var, y_var],
                "target": [0.0, 0.0],
            }
        elif x_var:
            task_metrics["position_centering"] = {
                "type": "raw_abs_inverse",
                "inputs": [x_var],
            }

        velocity_inputs = [x for x in [x_vel_var, y_vel_var] if x]
        if velocity_inputs:
            task_metrics["velocity_smoothness"] = {
                "type": "bounded_stability",
                "inputs": velocity_inputs,
                "scales": [0.5 for _ in velocity_inputs],
            }

        attitude_inputs = [x for x in [angle_var, ang_vel_var] if x]
        if attitude_inputs:
            task_metrics["stability"] = {
                "type": "bounded_stability",
                "inputs": attitude_inputs,
                "scales": [0.25 if x == angle_var else 0.5 for x in attitude_inputs],
            }
            task_metrics["attitude_smoothness"] = {
                "type": "raw_abs_inverse",
                "inputs": attitude_inputs,
            }

        if action_names:
            task_metrics["energy_cost"] = {"type": "action_cost"}

        if "both_contact" in events:
            task_metrics["contact_evidence"] = {"type": "event_score", "event": "both_contact"}
        elif "any_contact" in events:
            task_metrics["contact_evidence"] = {"type": "event_score", "event": "any_contact"}

        if "safe_contact" in events:
            task_metrics["safe_contact_score"] = {"type": "event_score", "event": "safe_contact"}

        if "stable_landing_condition" in events:
            task_metrics["success"] = {"type": "event_success", "event": "stable_landing_condition"}
            task_metrics["stable_landing_score"] = {"type": "event_score", "event": "stable_landing_condition"}
        elif "safe_contact" in events:
            task_metrics["success"] = {"type": "event_success", "event": "safe_contact"}

        progress_metrics = [
            name for name in [
                "approach_region_score",
                "position_centering",
                "velocity_smoothness",
                "stability",
                "contact_evidence",
            ]
            if name in task_metrics
        ]
        if progress_metrics:
            task_metrics["progress"] = {"type": "metric_mean", "metrics": progress_metrics}

        landing_quality_metrics = [
            name for name in [
                "approach_region_score",
                "velocity_smoothness",
                "stability",
                "contact_evidence",
                "safe_contact_score",
                "stable_landing_score",
                "success",
            ]
            if name in task_metrics
        ]
        if landing_quality_metrics:
            task_metrics["landing_quality"] = {"type": "metric_mean", "metrics": landing_quality_metrics}

        return {
            "source": "primitive_interface_generated_runtime_spec",
            "input_boundary": primitive_interface.get("input_boundary", "primitive_interface_conditioned"),
            "identity_hidden_from_llm": bool(primitive_interface.get("identity_hidden_from_llm", False)),
            "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
            "eureka_like_input_status": primitive_interface.get("env_code_parser", "planned_not_current"),
            "observation_mapping": observation_mapping,
            "action_variables": action_variables,
            "action_mapping": primitive_interface.get("action_mapping", {}),
            "events": events,
            "task_metrics": task_metrics,
        }

    @staticmethod
    def _write_json'''

new_text, n = pattern.subn(replacement, text)
if n != 1:
    raise SystemExit(f"Expected to replace exactly one _build_runtime_spec_from_primitive_interface block, replaced {n}")

p.write_text(new_text, encoding="utf-8")
PY

echo "[4/7] Patch edit_prompt.py: dynamic AST grammar/examples from allowed variables..."
cat > eg_rsa/llm/edit_prompt.py <<'PY'
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
PY

echo "[5/7] Patch runner.py: carry task description and repair invalid edit plans..."
python - <<'PY'
from pathlib import Path

p = Path("eg_rsa/runner.py")
text = p.read_text(encoding="utf-8")

old = '''        task_description = self._load_task_description()
        self._write_json(self.output_dir / "experiment_mode.json", self.mode.to_dict())
'''
new = '''        task_description = self._load_task_description()
        self.current_task_description = task_description
        self._write_json(self.output_dir / "experiment_mode.json", self.mode.to_dict())
'''
if old not in text:
    raise SystemExit("Could not find task_description assignment block in runner.py")
text = text.replace(old, new, 1)

old_func = '''    def _validate_evaluate_and_gate_edit_plan(self, schema: RewardSchema, raw_edit_plan: List[Dict[str, Any]], diagnostic_report: Dict[str, Any], trajectories: List[Dict[str, Any]], edit_decision: str, next_action: str, plan_metadata: Optional[Dict[str, Any]] = None):
        if edit_decision in {"no_edit", "need_more_evidence"} or raw_edit_plan == []:
            validation = EditPlanValidator.validate(schema, [], structural_context=self.structural_context)
            validation.errors.append(f"LLM chose {edit_decision} with next_action={next_action}; no schema edit applied.")
            return [], validation, None, None, next_action
        plan_metadata = plan_metadata or {}
        is_atomic = plan_metadata.get("atomicity") == "atomic" and plan_metadata.get("plan_type") == "coupled_rebalancing"
        validation = EditPlanValidator.validate(schema, raw_edit_plan, structural_context=self.structural_context)
        candidate_result = None
        gate_result = None
        edit_plan: List[Dict[str, Any]] = []
        if is_atomic and validation.rejected_edits:
            validation.errors.append("Atomic coupled package rejected before execution because at least one edit failed validation; partial execution is forbidden.")
            return [], validation, None, None, "structural_search"
        if validation.valid_edits:
            candidate_result = RewardCandidateEvaluator.evaluate(validation.valid_edits, trajectories, self.config.get("candidate_evaluator", {}))
            if is_atomic and len(candidate_result.accepted_edits) != len(validation.valid_edits):
                validation.errors.extend(candidate_result.warnings)
                validation.errors.append("Atomic coupled package rejected because candidate evaluation removed part of the package; partial execution is forbidden.")
                return [], validation, candidate_result, None, "structural_search"
            if not candidate_result.accepted_edits:
                validation.errors.extend(candidate_result.warnings)
                return [], validation, candidate_result, None, "structural_search"
            gate_result = EditDecisionGate.apply(schema, candidate_result.accepted_edits, diagnostic_report, self.config.get("edit_gate", {}), plan_metadata)
            edit_plan = gate_result.accepted_edits
            if is_atomic and len(edit_plan) != len(candidate_result.accepted_edits):
                validation.errors.extend(gate_result.warnings)
                validation.errors.append("Atomic coupled package rejected because gate removed part of the package; partial execution is forbidden.")
                return [], validation, candidate_result, gate_result, "structural_search"
            next_action = "apply_edit" if edit_plan else "structural_search"
            if not edit_plan:
                validation.errors.extend(gate_result.warnings)
        elif diagnostic_report.get("diagnostics", {}).get("dominant_component") and not self.mode.use_llm_edit:
            edit_plan = EditPlanValidator.safe_fallback(schema, diagnostic_report.get("diagnostics", {}))
            next_action = "apply_edit"
            validation.errors.append("No valid edit remained; used non-LLM safe fallback edit plan.")
        else:
            validation.errors.append("No valid edit remained; skipped schema edit.")
        if not self.mode.use_operator_constraints:
            validation.errors.append("Operator constraints disabled for ablation; no schema edit was applied.")
            edit_plan = []
        return edit_plan, validation, candidate_result, gate_result, next_action
'''

new_func = '''    def _validate_evaluate_and_gate_edit_plan(self, schema: RewardSchema, raw_edit_plan: List[Dict[str, Any]], diagnostic_report: Dict[str, Any], trajectories: List[Dict[str, Any]], edit_decision: str, next_action: str, plan_metadata: Optional[Dict[str, Any]] = None):
        if edit_decision in {"no_edit", "need_more_evidence"} or raw_edit_plan == []:
            validation = EditPlanValidator.validate(schema, [], structural_context=self.structural_context)
            validation.errors.append(f"LLM chose {edit_decision} with next_action={next_action}; no schema edit applied.")
            return [], validation, None, None, next_action

        plan_metadata = plan_metadata or {}
        is_atomic = plan_metadata.get("atomicity") == "atomic" and plan_metadata.get("plan_type") == "coupled_rebalancing"
        validation = EditPlanValidator.validate(schema, raw_edit_plan, structural_context=self.structural_context)

        # Repair invalid edit plans before wasting a full training iteration.
        # This is different from scale/behavior-audit repair: it fixes syntax,
        # variable-contract, operator, and AST validation failures.
        repair_cfg = self.config.get("edit_repair", {}) or {}
        repair_enabled = bool(repair_cfg.get("on_validation_error", True))
        max_repair_attempts = int(repair_cfg.get("max_attempts", 1) or 1)

        if (
            self.mode.use_llm_edit
            and repair_enabled
            and validation.rejected_edits
            and max_repair_attempts > 0
        ):
            original_validation_dict = validation.to_dict()
            current_plan = list(raw_edit_plan or [])
            current_errors = list(validation.errors or [])

            for repair_attempt in range(1, max_repair_attempts + 1):
                try:
                    failed_edit_response = {
                        "repair_trigger": "edit_plan_validation_error",
                        "attempt": repair_attempt,
                        "previous_edit_plan": current_plan,
                        "validation": original_validation_dict,
                        "latest_validation_errors": current_errors,
                        "allowed_formula_variables": list(
                            (schema.metadata or {}).get(
                                "allowed_formula_variables",
                                self.structural_context.get("allowed_formula_variables", []),
                            )
                        ),
                        "instruction": (
                            "Repair the edit_plan so every AST variable uses only allowed_formula_variables. "
                            "Preserve the intent if possible; otherwise return no_edit + continue_training."
                        ),
                    }

                    repair_response = self.edit_agent.generate_repair_edit_plan(
                        task_description=getattr(self, "current_task_description", "") or self._load_task_description(),
                        current_reward_schema=schema.to_dict(),
                        diagnostic_report=diagnostic_report,
                        retrieved_memories=[],
                        retrieved_lessons=[],
                        reflection_report={"strategy": plan_metadata.get("reflection_strategy", {})},
                        failed_edit_response=failed_edit_response,
                        scale_audit_report={"audit_pass": True, "source": "validation_repair_not_scale_audit"},
                        behavior_risk_report={"audit_pass": True, "source": "validation_repair_not_behavior_audit"},
                    )

                    repaired_plan = repair_response.get("edit_plan", [])
                    repaired_metadata = self._extract_plan_metadata(
                        repair_response,
                        {"strategy": plan_metadata.get("reflection_strategy", {})},
                    )
                    repaired_validation = EditPlanValidator.validate(
                        schema,
                        repaired_plan,
                        structural_context=self.structural_context,
                    )

                    repaired_validation.warnings.append(
                        "Edit plan repaired after validation failure. "
                        f"attempt={repair_attempt}; original_errors={current_errors}"
                    )

                    if repaired_validation.valid_edits and not repaired_validation.rejected_edits:
                        raw_edit_plan = repaired_plan
                        validation = repaired_validation
                        plan_metadata = repaired_metadata
                        is_atomic = (
                            plan_metadata.get("atomicity") == "atomic"
                            and plan_metadata.get("plan_type") == "coupled_rebalancing"
                        )
                        next_action = self._extract_next_action(repair_response)
                        break

                    current_plan = repaired_plan
                    current_errors = list(repaired_validation.errors or ["repair produced no valid edits"])
                    validation.warnings.append(
                        f"Validation repair attempt {repair_attempt} failed: {current_errors}"
                    )

                except Exception as exc:
                    validation.warnings.append(
                        f"Validation repair attempt {repair_attempt} raised {type(exc).__name__}: {exc}"
                    )

        candidate_result = None
        gate_result = None
        edit_plan: List[Dict[str, Any]] = []

        if is_atomic and validation.rejected_edits:
            validation.errors.append("Atomic coupled package rejected before execution because at least one edit failed validation; partial execution is forbidden.")
            return [], validation, None, None, "structural_search"

        if validation.valid_edits:
            candidate_result = RewardCandidateEvaluator.evaluate(validation.valid_edits, trajectories, self.config.get("candidate_evaluator", {}))
            if is_atomic and len(candidate_result.accepted_edits) != len(validation.valid_edits):
                validation.errors.extend(candidate_result.warnings)
                validation.errors.append("Atomic coupled package rejected because candidate evaluation removed part of the package; partial execution is forbidden.")
                return [], validation, candidate_result, None, "structural_search"
            if not candidate_result.accepted_edits:
                validation.errors.extend(candidate_result.warnings)
                return [], validation, candidate_result, None, "structural_search"
            gate_result = EditDecisionGate.apply(schema, candidate_result.accepted_edits, diagnostic_report, self.config.get("edit_gate", {}), plan_metadata)
            edit_plan = gate_result.accepted_edits
            if is_atomic and len(edit_plan) != len(candidate_result.accepted_edits):
                validation.errors.extend(gate_result.warnings)
                validation.errors.append("Atomic coupled package rejected because gate removed part of the package; partial execution is forbidden.")
                return [], validation, candidate_result, gate_result, "structural_search"
            next_action = "apply_edit" if edit_plan else "structural_search"
            if not edit_plan:
                validation.errors.extend(gate_result.warnings)
        elif diagnostic_report.get("diagnostics", {}).get("dominant_component") and not self.mode.use_llm_edit:
            edit_plan = EditPlanValidator.safe_fallback(schema, diagnostic_report.get("diagnostics", {}))
            next_action = "apply_edit"
            validation.errors.append("No valid edit remained; used non-LLM safe fallback edit plan.")
        else:
            validation.errors.append("No valid edit remained after validation and optional repair; skipped schema edit.")

        if not self.mode.use_operator_constraints:
            validation.errors.append("Operator constraints disabled for ablation; no schema edit was applied.")
            edit_plan = []

        return edit_plan, validation, candidate_result, gate_result, next_action
'''

if old_func not in text:
    raise SystemExit("Could not find exact _validate_evaluate_and_gate_edit_plan block in runner.py")

text = text.replace(old_func, new_func, 1)
p.write_text(text, encoding="utf-8")
PY

echo "[6/7] Patch long experiment script: fix primitive_interface_path..."
python - <<'PY'
from pathlib import Path

p = Path("run_landing_10iter_6seeds_2M.sh")
if not p.exists():
    print("[WARN] run_landing_10iter_6seeds_2M.sh not found; skipped script patch.")
    raise SystemExit(0)

text = p.read_text(encoding="utf-8")

if 'INTERFACE_PATH="${BASE_EXP}/interface/generated_primitive_interface.json"' not in text:
    text = text.replace(
        'DIAG_PATH="${BASE_EXP}/bootstrap/generated_diagnostics.yml"\n',
        'DIAG_PATH="${BASE_EXP}/bootstrap/generated_diagnostics.yml"\n'
        'INTERFACE_PATH="${BASE_EXP}/interface/generated_primitive_interface.json"\n',
        1,
    )

if 'if [ ! -f "${INTERFACE_PATH}" ]; then' not in text:
    text = text.replace(
        'if [ ! -f "${DIAG_PATH}" ]; then\n'
        '  echo "[ERROR] Missing diagnostics: ${DIAG_PATH}"\n'
        '  exit 1\n'
        'fi\n',
        'if [ ! -f "${DIAG_PATH}" ]; then\n'
        '  echo "[ERROR] Missing diagnostics: ${DIAG_PATH}"\n'
        '  exit 1\n'
        'fi\n'
        '\n'
        'if [ ! -f "${INTERFACE_PATH}" ]; then\n'
        '  echo "[ERROR] Missing primitive interface: ${INTERFACE_PATH}"\n'
        '  exit 1\n'
        'fi\n',
        1,
    )

if 'echo "INTERFACE_PATH   = ${INTERFACE_PATH}"' not in text:
    text = text.replace(
        'echo "DIAG_PATH        = ${DIAG_PATH}"\n',
        'echo "DIAG_PATH        = ${DIAG_PATH}"\n'
        'echo "INTERFACE_PATH   = ${INTERFACE_PATH}"\n',
        1,
    )

# Add primitive_interface_path under schema_source in generated YAML.
if 'primitive_interface_path: ${INTERFACE_PATH}' not in text:
    text = text.replace(
        '  schema_source:\n'
        '    type: manual\n'
        '    initial_schema_path: ${SCHEMA_PATH}\n',
        '  schema_source:\n'
        '    type: manual\n'
        '    initial_schema_path: ${SCHEMA_PATH}\n'
        '    primitive_interface_path: ${INTERFACE_PATH}\n',
        1,
    )

# Add config toggles for validation repair.
if 'edit_repair:' not in text:
    text = text.replace(
        'edit_gate:\n'
        '  max_edits_per_iteration: 3\n'
        '  min_target_ratio: 0.02\n'
        '  min_target_trigger_rate: 0.01\n',
        'edit_gate:\n'
        '  max_edits_per_iteration: 3\n'
        '  min_target_ratio: 0.02\n'
        '  min_target_trigger_rate: 0.01\n'
        '\n'
        'edit_repair:\n'
        '  on_validation_error: true\n'
        '  max_attempts: 1\n',
        1,
    )

p.write_text(text, encoding="utf-8")
PY

echo "[7/7] Syntax checks..."
python -m py_compile \
  eg_rsa/schema_sources/llm_bootstrap.py \
  eg_rsa/llm/edit_prompt.py \
  eg_rsa/reward/edit_plan_validator.py \
  eg_rsa/runner.py

if [ -f run_landing_10iter_6seeds_2M.sh ]; then
  bash -n run_landing_10iter_6seeds_2M.sh
fi

echo ""
echo "Patch done."
echo "Backups saved in: ${BACKUP_DIR}"
echo ""
echo "Recommended next steps:"
echo "  1) Stop old processes if any:"
echo "       ps aux | grep -E 'train_eg_rsa|run_landing_10iter' | grep -v grep"
echo ""
echo "  2) Backup old incomplete experiment:"
echo "       mv experiments/landing_v2_1_10iter_6seeds_2M experiments/landing_v2_1_10iter_6seeds_2M.bak_${TS} 2>/dev/null || true"
echo ""
echo "  3) Rebuild the bootstrap check once, so generated_diagnostics.yml includes safe/stable landing metrics:"
echo "       rm -rf experiments/eg_rsa_landing_v2_1_source_aware_bootstrap_check"
echo "       python train_eg_rsa.py --config configs/eg_rsa_landing_v2_1_source_aware_bootstrap_check.yml"
echo ""
echo "  4) Restart long experiment:"
echo "       mkdir -p experiments/landing_v2_1_10iter_6seeds_2M"
echo "       nohup env TOTAL_TIMESTEPS=2000000 N_ENVS=16 ./run_landing_10iter_6seeds_2M.sh > experiments/landing_v2_1_10iter_6seeds_2M/nohup.log 2>&1 &"
