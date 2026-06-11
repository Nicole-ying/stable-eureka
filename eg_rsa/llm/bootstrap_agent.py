from __future__ import annotations

import json
from typing import Any, Dict, Optional

from eg_rsa.llm.json_parser import extract_json_object


class BootstrapAgent:
    """Generate an initial reward schema from a primitive-only task interface.

    V2 requires the LLM to first produce an explicit reward_blueprint, then
    compile that blueprint into a safe formula-based reward schema.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.last_prompt: str = ""
        self.last_response_text: str = ""

    def generate_bootstrap(
        self,
        primitive_interface: Dict[str, Any],
        task_description: str = "",
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(primitive_interface, task_description)
        self.last_prompt = prompt

        if self.llm_client is None:
            result = self._fallback_bootstrap(primitive_interface, task_description)
            self.last_response_text = json.dumps(result, indent=2, ensure_ascii=False)
            return result

        response_text = self.llm_client.generate(prompt)
        self.last_response_text = response_text
        parsed = extract_json_object(response_text)
        return self._normalize(parsed, primitive_interface)

    @staticmethod
    def _build_prompt(primitive_interface: Dict[str, Any], task_description: str) -> str:
        allowed_vars = primitive_interface.get("allowed_formula_variables", [])
        allowed_funcs = primitive_interface.get("allowed_formula_functions", [])
        semantic_roles = primitive_interface.get("semantic_roles", [])
        allowed_component_types = primitive_interface.get(
            "allowed_component_types_v2",
            ["formula_component", "conditional_formula_component", "event_predicate", "action_penalty"],
        )

        output_shape = {
            "reward_blueprint": {
                "task_objective": "one-sentence objective inferred from the task text",
                "primitive_variable_roles": {
                    "progress_variables": [],
                    "safety_variables": [],
                    "terminal_evidence_variables": [],
                    "control_variables": []
                },
                "phase_structure": [
                    {
                        "phase": "approach_or_progress",
                        "purpose": "encourage measurable progress toward task completion",
                        "reward_intent": "what signal should exist in this phase"
                    },
                    {
                        "phase": "controlled_execution",
                        "purpose": "encourage safe/controlled behavior while progress is happening",
                        "reward_intent": "what signal should exist in this phase"
                    },
                    {
                        "phase": "completion",
                        "purpose": "reward terminal completion evidence once",
                        "reward_intent": "what terminal evidence should trigger success"
                    }
                ],
                "component_blueprint": [
                    {
                        "name": "r_progress_guidance",
                        "role": "dense_guidance",
                        "phase": "approach_or_progress",
                        "design_intent": "progress signal, not passive state maintenance",
                        "anti_exploit_note": "why this cannot be maximized forever without task progress"
                    }
                ],
                "anti_exploit_principles": [
                    "Dense reward should not be maximized indefinitely without progress.",
                    "Passive stability/alignment should support progress rather than replace progress.",
                    "Action cost should penalize effort magnitude and never reward a signed action accidentally."
                ]
            },
            "initial_schema": {
                "version": 2,
                "metadata": {
                    "source": "llm_bootstrap",
                    "task": "..."
                },
                "components": [],
                "event_rules": []
            },
            "diagnostics": {
                "expected_failure_modes": [],
                "risk_notes": []
            },
            "bootstrap_report": {
                "design_rationale": "...",
                "assumptions": [],
                "risk_notes": [],
                "primitive_interface_only": True
            }
        }

        structure_few_shot = {
            "example_blueprint_only": {
                "task_type": "goal-directed control task",
                "reward_structure": [
                    {
                        "role": "progress_guidance",
                        "principle": "encourage measurable movement toward completion, not passive state maintenance"
                    },
                    {
                        "role": "control_quality",
                        "principle": "encourage safe/smooth behavior while the agent is making progress"
                    },
                    {
                        "role": "terminal_success",
                        "principle": "reward primitive terminal evidence once, not repeatedly"
                    },
                    {
                        "role": "control_cost",
                        "principle": "penalize action magnitude; signed actions must not accidentally become positive reward"
                    }
                ]
            }
        }

        return f"""
You are the EG-RSA-V2 bootstrap agent.

You receive a primitive-only reinforcement-learning task interface.
You are not given any previous reward schema, previous diagnostic metric definitions,
previous event predicate definitions, or official environment reward feedback.

Your job has two steps:
1. Infer a reward design blueprint from the task text and primitive variables.
2. Compile that blueprint into a safe formula-based reward schema.

Task description:
{task_description or primitive_interface.get("task_description", "")}

Primitive interface:
{json.dumps(primitive_interface, indent=2, ensure_ascii=False)}

Allowed formula variables:
{json.dumps(allowed_vars, ensure_ascii=False)}

Allowed formula functions:
{json.dumps(allowed_funcs, ensure_ascii=False)}

Allowed schema item types:
{json.dumps(allowed_component_types, ensure_ascii=False)}

Allowed semantic roles:
{json.dumps(semantic_roles, ensure_ascii=False)}

Environment-agnostic reward design protocol:
1. First identify which primitive variables can express progress, safety/control, terminal evidence, and control effort.
2. Do not jump directly from primitive variables to formulas. Explain the reward blueprint first.
3. A goal-directed control task usually needs progress guidance, control quality, terminal success, and control cost.
4. Dense rewards should be progress-aligned. A policy should not receive high dense reward forever by merely staying stable, aligned, or stationary far from completion.
5. Stability/alignment rewards are useful only when they support progress or completion; they should not replace progress.
6. Conditional final-phase rewards are allowed, but the schema should also contain a process/progress signal that is not only available at the final instant.
7. Terminal success should be sparse, one-time, and based only on primitive terminal evidence.
8. Action penalties must represent effort magnitude. If an action variable can be signed, the penalty expression must not become positive reward for one action direction.
9. Do not use hidden metrics or invented event names. Every formula must use only allowed primitive variables and allowed functions.
10. Schema language boundary:
    - formula_component.formula is a primitive numeric expression.
    - conditional_formula_component.condition is a primitive boolean expression.
    - event_predicate.condition.expression is a primitive boolean expression.
    - Do NOT put event_predicate.expression at the top level of the event rule.
    - Do NOT reference an event rule by name inside a formula or component condition.
    - If you need a terminal completion reward, use event_rules with type="event_predicate" rather than a conditional_formula_component that points to an event name.
11. Output JSON only. Do not output markdown.

Structure-only few-shot example. This is not a reward function to copy:
{json.dumps(structure_few_shot, indent=2, ensure_ascii=False)}

Required output JSON shape:
{json.dumps(output_shape, indent=2, ensure_ascii=False)}

Generate a compact schema with 3-7 components and 1-3 event rules.
""".strip()

    @staticmethod
    def _normalize(parsed: Dict[str, Any], primitive_interface: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            raise ValueError("Bootstrap LLM response must parse to a JSON object")

        if "initial_schema" not in parsed:
            raise ValueError("Bootstrap response missing initial_schema")

        blueprint = parsed.get("reward_blueprint")
        if not isinstance(blueprint, dict):
            parsed["reward_blueprint"] = {
                "task_objective": "missing_blueprint",
                "primitive_variable_roles": {},
                "phase_structure": [],
                "component_blueprint": [],
                "anti_exploit_principles": [],
            }

        schema = dict(parsed["initial_schema"] or {})
        schema.setdefault("version", 2)
        schema.setdefault("metadata", {})
        schema.setdefault("components", [])
        schema.setdefault("event_rules", [])
        schema["metadata"].setdefault("source", "llm_bootstrap")
        schema["metadata"].setdefault("reward_blueprint_present", True)

        normalization_notes = []

        # Normalize event rules first so component conditions may expand
        # event-name aliases into primitive expressions.
        event_expr_by_name: Dict[str, str] = {}
        for rule in schema.get("event_rules", []):
            if isinstance(rule, dict):
                notes = BootstrapAgent._normalize_event_rule(rule)
                normalization_notes.extend(notes)
                expr = BootstrapAgent._event_rule_expression(rule)
                if rule.get("name") and expr:
                    event_expr_by_name[str(rule["name"])] = str(expr)

        for component in schema.get("components", []):
            if isinstance(component, dict):
                notes = BootstrapAgent._normalize_component(component, event_expr_by_name)
                normalization_notes.extend(notes)

        if normalization_notes:
            parsed.setdefault("bootstrap_report", {})
            parsed["bootstrap_report"].setdefault("normalization_notes", [])
            parsed["bootstrap_report"]["normalization_notes"].extend(normalization_notes)
            schema["metadata"].setdefault("normalization_notes", [])
            schema["metadata"]["normalization_notes"].extend(normalization_notes)

        parsed["initial_schema"] = schema
        parsed.setdefault("diagnostics", {})
        parsed.setdefault("bootstrap_report", {})
        parsed["bootstrap_report"].setdefault("primitive_interface_only", True)
        parsed["bootstrap_report"].setdefault("primitive_env", primitive_interface.get("env"))
        return parsed

    @staticmethod
    def _normalize_component(component: Dict[str, Any], event_expr_by_name: Dict[str, str] | None = None) -> list[str]:
        event_expr_by_name = event_expr_by_name or {}
        notes: list[str] = []

        ctype = component.get("type")
        component.setdefault("enabled", True)
        component.setdefault("weight", 1.0)
        component.setdefault("params", {})

        params = dict(component.get("params", {}) or {})

        if "formula" in component:
            params["formula"] = component["formula"]
        elif "formula" in params:
            component["formula"] = params["formula"]

        if ctype == "conditional_formula_component":
            if "condition" in component:
                params["condition"] = component["condition"]
            elif "condition" in params:
                component["condition"] = params["condition"]

            cond = params.get("condition")

            # If the LLM wrote condition=<event_rule_name>, expand it into the
            # event rule's primitive expression. This keeps the saved schema
            # primitive-only and avoids introducing event-name variables.
            if isinstance(cond, str):
                stripped = cond.strip()
                if stripped in event_expr_by_name:
                    expanded = event_expr_by_name[stripped]
                    params["condition"] = expanded
                    component["condition"] = expanded
                    notes.append(
                        f"Expanded component {component.get('name')} condition event alias "
                        f"{stripped!r} into primitive expression."
                    )

        component["params"] = params
        return notes

    @staticmethod
    def _normalize_event_rule(rule: Dict[str, Any]) -> list[str]:
        notes: list[str] = []

        rule.setdefault("enabled", True)
        rule.setdefault("weight", 1.0)
        rule.setdefault("one_time", True)

        condition = rule.get("condition", {})

        if isinstance(condition, str):
            rule["condition"] = {
                "expression": condition,
                "duration_steps": int(rule.get("duration_steps", 1) or 1),
            }
            return notes

        if not isinstance(condition, dict):
            condition = {}

        condition = dict(condition)

        # Common LLM mistake: putting expression/formula at the event-rule top
        # level rather than inside condition.expression.
        top_expr = rule.get("expression") or rule.get("formula")
        if top_expr and not (condition.get("expression") or condition.get("formula")):
            condition["expression"] = str(top_expr)
            notes.append(
                f"Moved top-level expression/formula of event rule {rule.get('name')} "
                "into condition.expression."
            )

        condition.setdefault(
            "duration_steps",
            int(rule.get("duration_steps", condition.get("duration_steps", 1)) or 1),
        )

        rule["condition"] = condition

        # Remove duplicate top-level fields to keep the persisted schema clean.
        rule.pop("expression", None)
        rule.pop("formula", None)
        rule.pop("duration_steps", None)

        return notes

    @staticmethod
    def _event_rule_expression(rule: Dict[str, Any]) -> str | None:
        condition = rule.get("condition", {})
        if isinstance(condition, str):
            return condition
        if isinstance(condition, dict):
            expr = condition.get("expression") or condition.get("formula")
            return str(expr) if expr else None
        return None

    @staticmethod
    def _fallback_bootstrap(
        primitive_interface: Dict[str, Any],
        task_description: str = "",
    ) -> Dict[str, Any]:
        task = task_description or primitive_interface.get("task_description", "")

        blueprint = {
            "task_objective": task or "Goal-directed control task",
            "primitive_variable_roles": {
                "progress_variables": ["x", "y", "vx", "vy"],
                "safety_variables": ["vx", "vy", "angle", "angular_velocity"],
                "terminal_evidence_variables": ["left_contact", "right_contact", "vx", "vy", "angle"],
                "control_variables": ["main_engine", "side_engine"],
            },
            "phase_structure": [
                {
                    "phase": "approach_or_progress",
                    "purpose": "provide a broad process signal toward completion",
                    "reward_intent": "encourage reducing task-relevant position/velocity error using primitive variables",
                },
                {
                    "phase": "controlled_execution",
                    "purpose": "keep motion safe while making progress",
                    "reward_intent": "encourage moderate speed and stable attitude",
                },
                {
                    "phase": "completion",
                    "purpose": "reward primitive terminal evidence once",
                    "reward_intent": "reward simultaneous contact and controlled state",
                },
            ],
            "component_blueprint": [
                {
                    "name": "r_progress_guidance",
                    "role": "dense_guidance",
                    "phase": "approach_or_progress",
                    "design_intent": "broad progress-aligned dense signal",
                    "anti_exploit_note": "does not rely solely on passive stability",
                },
                {
                    "name": "r_control_quality",
                    "role": "stability_quality",
                    "phase": "controlled_execution",
                    "design_intent": "safe smooth control during approach",
                    "anti_exploit_note": "weighted below progress guidance",
                },
                {
                    "name": "r_terminal_success",
                    "role": "terminal_success",
                    "phase": "completion",
                    "design_intent": "one-time primitive terminal evidence",
                    "anti_exploit_note": "one_time prevents repeated event farming",
                },
            ],
            "anti_exploit_principles": [
                "Dense rewards should not be maximized indefinitely without progress.",
                "Control cost is based on action magnitudes.",
            ],
        }

        schema = {
            "version": 2,
            "metadata": {
                "source": "fallback_bootstrap",
                "task": task,
                "reward_blueprint_present": True,
            },
            "components": [
                {
                    "name": "r_progress_guidance",
                    "type": "formula_component",
                    "weight": 0.8,
                    "formula": "1.0 - min(abs(x) + 0.5 * abs(vx) + 0.5 * abs(vy), 1.0)",
                    "params": {"formula": "1.0 - min(abs(x) + 0.5 * abs(vx) + 0.5 * abs(vy), 1.0)"},
                    "clip": [0.0, 1.0],
                    "enabled": True,
                    "semantic_role": "dense_guidance",
                    "reward_timing": "dense",
                    "behavior_channel": "progress",
                },
                {
                    "name": "r_attitude_control",
                    "type": "formula_component",
                    "weight": 0.3,
                    "formula": "1.0 - min(abs(angle) + abs(angular_velocity), 1.0)",
                    "params": {"formula": "1.0 - min(abs(angle) + abs(angular_velocity), 1.0)"},
                    "clip": [0.0, 1.0],
                    "enabled": True,
                    "semantic_role": "stability_quality",
                    "reward_timing": "dense",
                    "behavior_channel": "attitude",
                },
                {
                    "name": "r_control_cost",
                    "type": "action_penalty",
                    "weight": 0.05,
                    "formula": "-(main_engine + abs(side_engine))",
                    "params": {"formula": "-(main_engine + abs(side_engine))"},
                    "clip": [-1.0, 0.0],
                    "enabled": True,
                    "semantic_role": "control_cost",
                    "reward_timing": "dense",
                    "behavior_channel": "action",
                },
            ],
            "event_rules": [
                {
                    "name": "r_primitive_terminal_success",
                    "type": "event_predicate",
                    "weight": 60.0,
                    "condition": {
                        "expression": "left_contact and right_contact and abs(vy) < 0.4 and abs(vx) < 0.4 and abs(angle) < 0.4",
                        "duration_steps": 1,
                    },
                    "one_time": True,
                    "enabled": True,
                    "semantic_role": "terminal_success",
                    "reward_timing": "sparse_event",
                    "behavior_channel": "completion",
                }
            ],
        }

        return {
            "reward_blueprint": blueprint,
            "initial_schema": schema,
            "diagnostics": {
                "expected_failure_modes": [
                    "dense reward may still be exploited without terminal completion",
                    "terminal event may be sparse early in training",
                ],
                "risk_notes": [
                    "Fallback schema is intentionally generic and should be improved by EG-RSA iterations."
                ],
            },
            "bootstrap_report": {
                "design_rationale": "Fallback bootstrap follows a primitive-only blueprint with progress, control quality, action cost, and one-time terminal evidence.",
                "assumptions": [
                    "The primitive interface exposes position, velocity, attitude, contact, and action variables."
                ],
                "risk_notes": [
                    "This fallback is conservative and not intended as a final reward."
                ],
                "primitive_interface_only": True,
            },
        }
