from __future__ import annotations

import json
from typing import Any, Dict, Optional

from eg_rsa.llm.json_parser import extract_json_object


class BootstrapAgent:
    """Generate initial AST-first reward schema from primitive-only task interface."""

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
    def _ast_grammar() -> Dict[str, Any]:
        return {
            "leaf_nodes": [
                {"var": "x"},
                {"const": 0.5},
                {"bool": True},
            ],
            "numeric_ops": [
                {"op": "add", "args": [{"var": "x"}, {"var": "y"}]},
                {"op": "sub", "left": {"const": 1.0}, "right": {"var": "x"}},
                {"op": "mul", "args": [{"const": 0.5}, {"op": "abs", "arg": {"var": "vx"}}]},
                {"op": "div", "left": {"var": "x"}, "right": {"const": 2.0}},
                {"op": "neg", "arg": {"var": "main_engine"}},
                {"op": "abs", "arg": {"var": "angle"}},
                {"op": "min", "args": [{"var": "x"}, {"const": 1.0}]},
                {"op": "max", "args": [{"var": "x"}, {"const": 0.0}]},
                {"op": "clip", "args": [{"var": "x"}, {"const": -1.0}, {"const": 1.0}]}
            ],
            "boolean_ops": [
                {"op": "and", "args": [{"var": "left_contact"}, {"var": "right_contact"}]},
                {"op": "or", "args": [{"var": "left_contact"}, {"var": "right_contact"}]},
                {"op": "not", "arg": {"var": "left_contact"}},
                {"op": "lt", "left": {"op": "abs", "arg": {"var": "vy"}}, "right": {"const": 0.4}},
                {"op": "gt", "left": {"op": "abs", "arg": {"var": "angle"}}, "right": {"const": 0.6}}
            ],
        }

    @staticmethod
    def _build_prompt(primitive_interface: Dict[str, Any], task_description: str) -> str:
        allowed_vars = primitive_interface.get("allowed_formula_variables", [])
        semantic_roles = primitive_interface.get("semantic_roles", [])

        output_shape = {
            "reward_blueprint": {
                "task_objective": "one-sentence objective inferred from task text",
                "primitive_variable_roles": {
                    "progress_variables": [],
                    "safety_variables": [],
                    "terminal_evidence_variables": [],
                    "control_variables": [],
                },
                "phase_structure": [],
                "component_blueprint": [],
                "anti_exploit_principles": [],
            },
            "initial_schema": {
                "version": 2,
                "metadata": {
                    "source": "llm_bootstrap_ast",
                    "formula_ir": "ast",
                },
                "components": [
                    {
                        "name": "r_progress_guidance",
                        "type": "formula_component",
                        "weight": 1.0,
                        "formula_ast": {"op": "sub", "left": {"const": 1.0}, "right": {"op": "min", "args": [{"op": "abs", "arg": {"var": "x"}}, {"const": 1.0}]}},
                        "params": {
                            "formula_ast": {"op": "sub", "left": {"const": 1.0}, "right": {"op": "min", "args": [{"op": "abs", "arg": {"var": "x"}}, {"const": 1.0}]}}
                        },
                        "clip": [0.0, 1.0],
                        "enabled": True,
                        "semantic_role": "dense_guidance",
                        "reward_timing": "dense",
                        "behavior_channel": "progress"
                    }
                ],
                "event_rules": [
                    {
                        "name": "r_terminal_success",
                        "type": "event_predicate",
                        "weight": 50.0,
                        "condition": {
                            "expr_ast": {"op": "and", "args": [{"var": "left_contact"}, {"var": "right_contact"}]},
                            "duration_steps": 1
                        },
                        "one_time": True,
                        "enabled": True,
                        "semantic_role": "terminal_success",
                        "reward_timing": "sparse_event",
                        "behavior_channel": "completion"
                    }
                ],
            },
            "diagnostics": {
                "expected_failure_modes": [],
                "risk_notes": [],
            },
            "bootstrap_report": {
                "design_rationale": "...",
                "assumptions": [],
                "risk_notes": [],
                "primitive_interface_only": True,
            },
        }

        return f"""
You are the EG-RSA-V2 AST bootstrap agent.

You receive a primitive-only RL task interface. You must output AST-first RewardSchema JSON.
Do not output Python code. Do not output string formulas. Do not output condition strings.

Task description:
{task_description or primitive_interface.get("task_description", "")}

Primitive interface:
{json.dumps(primitive_interface, indent=2, ensure_ascii=False)}

Allowed variables:
{json.dumps(allowed_vars, ensure_ascii=False)}

Allowed semantic roles:
{json.dumps(semantic_roles, ensure_ascii=False)}

AST grammar:
{json.dumps(BootstrapAgent._ast_grammar(), indent=2, ensure_ascii=False)}

Hard constraints:
1. Every formula_component must contain formula_ast and params.formula_ast.
2. Every conditional_formula_component must contain formula_ast, condition_ast, params.formula_ast, params.condition_ast.
3. Every event_predicate must contain condition.expr_ast.
4. Do not use fields named formula, expression, or string condition.
5. All AST variables must come from allowed variables.
6. Use compact schemas: 3-7 components and 1-3 event rules.
7. Use semantic_role for every component/event.
8. Action cost must be a formula_component with semantic_role="control_cost" and reward-signed negative formula_ast.
9. Output JSON only.

Required output shape:
{json.dumps(output_shape, indent=2, ensure_ascii=False)}
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
        schema["metadata"].setdefault("source", "llm_bootstrap_ast")
        schema["metadata"].setdefault("formula_ir", "ast")
        schema["metadata"].setdefault("reward_blueprint_present", True)
        parsed["initial_schema"] = schema
        parsed.setdefault("diagnostics", {})
        parsed.setdefault("bootstrap_report", {})
        parsed["bootstrap_report"].setdefault("primitive_interface_only", True)
        parsed["bootstrap_report"].setdefault("primitive_env", primitive_interface.get("env"))
        return parsed

    @staticmethod
    def _fallback_bootstrap(
        primitive_interface: Dict[str, Any],
        task_description: str = "",
    ) -> Dict[str, Any]:
        task = task_description or primitive_interface.get("task_description", "")

        def var(name):
            return {"var": name}

        def const(value):
            return {"const": value}

        def abs_(x):
            return {"op": "abs", "arg": x}

        def add(*args):
            return {"op": "add", "args": list(args)}

        def mul(*args):
            return {"op": "mul", "args": list(args)}

        def sub(left, right):
            return {"op": "sub", "left": left, "right": right}

        def neg(x):
            return {"op": "neg", "arg": x}

        def min_(*args):
            return {"op": "min", "args": list(args)}

        def lt(left, right):
            return {"op": "lt", "left": left, "right": right}

        def gt(left, right):
            return {"op": "gt", "left": left, "right": right}

        def and_(*args):
            return {"op": "and", "args": list(args)}

        def or_(*args):
            return {"op": "or", "args": list(args)}

        blueprint = {
            "task_objective": task or "Goal-directed control task",
            "primitive_variable_roles": {
                "progress_variables": ["x", "y", "vx", "vy"],
                "safety_variables": ["vx", "vy", "angle", "angular_velocity"],
                "terminal_evidence_variables": ["left_contact", "right_contact", "vx", "vy", "angle"],
                "control_variables": ["main_engine", "side_engine"],
            },
            "phase_structure": [
                {"phase": "approach_or_progress", "purpose": "provide dense process signal", "reward_intent": "center and slow the lander"},
                {"phase": "controlled_execution", "purpose": "keep attitude and speed safe", "reward_intent": "stable controlled approach"},
                {"phase": "completion", "purpose": "reward stable ground contact once", "reward_intent": "one-time success evidence"},
            ],
            "component_blueprint": [
                {"name": "r_progress_guidance", "role": "dense_guidance", "phase": "approach_or_progress"},
                {"name": "r_attitude_control", "role": "stability_quality", "phase": "controlled_execution"},
                {"name": "r_control_cost", "role": "control_cost", "phase": "controlled_execution"},
                {"name": "r_primitive_terminal_success", "role": "terminal_success", "phase": "completion"},
            ],
            "anti_exploit_principles": [
                "Dense reward should not replace terminal completion.",
                "Control cost is action magnitude penalty only.",
                "Terminal reward is one-time.",
            ],
        }

        progress_ast = sub(
            const(1.0),
            min_(
                add(abs_(var("x")), mul(const(0.5), abs_(var("vx"))), mul(const(0.5), abs_(var("vy")))),
                const(1.0),
            ),
        )
        attitude_ast = sub(
            const(1.0),
            min_(add(abs_(var("angle")), abs_(var("angular_velocity"))), const(1.0)),
        )
        control_ast = neg(add(var("main_engine"), abs_(var("side_engine"))))
        success_ast = and_(
            var("left_contact"),
            var("right_contact"),
            lt(abs_(var("vy")), const(0.4)),
            lt(abs_(var("vx")), const(0.4)),
            lt(abs_(var("angle")), const(0.4)),
        )
        crash_ast = and_(
            or_(var("left_contact"), var("right_contact")),
            or_(gt(abs_(var("vy")), const(0.8)), gt(abs_(var("angle")), const(0.7))),
        )

        schema = {
            "version": 2,
            "metadata": {
                "source": "fallback_bootstrap_ast",
                "formula_ir": "ast",
                "task": task,
                "reward_blueprint_present": True,
            },
            "components": [
                {
                    "name": "r_progress_guidance",
                    "type": "formula_component",
                    "weight": 0.8,
                    "formula_ast": progress_ast,
                    "params": {"formula_ast": progress_ast},
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
                    "formula_ast": attitude_ast,
                    "params": {"formula_ast": attitude_ast},
                    "clip": [0.0, 1.0],
                    "enabled": True,
                    "semantic_role": "stability_quality",
                    "reward_timing": "dense",
                    "behavior_channel": "attitude",
                },
                {
                    "name": "r_control_cost",
                    "type": "formula_component",
                    "weight": 0.05,
                    "formula_ast": control_ast,
                    "params": {"formula_ast": control_ast},
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
                    "condition": {"expr_ast": success_ast, "duration_steps": 1},
                    "one_time": True,
                    "enabled": True,
                    "semantic_role": "terminal_success",
                    "reward_timing": "sparse_event",
                    "behavior_channel": "completion",
                },
                {
                    "name": "r_crash_safety",
                    "type": "event_predicate",
                    "weight": -30.0,
                    "condition": {"expr_ast": crash_ast, "duration_steps": 1},
                    "one_time": True,
                    "enabled": True,
                    "semantic_role": "safety_constraint",
                    "reward_timing": "sparse_event",
                    "behavior_channel": "safety",
                },
            ],
        }

        return {
            "reward_blueprint": blueprint,
            "initial_schema": schema,
            "diagnostics": {
                "expected_failure_modes": [
                    "terminal event may be sparse early in training",
                    "dense progress may need later rebalancing",
                ],
                "risk_notes": [
                    "Fallback AST schema is conservative and should be improved by EG-RSA iterations."
                ],
            },
            "bootstrap_report": {
                "design_rationale": "Fallback AST bootstrap uses progress, attitude control, action cost, terminal success, and crash safety.",
                "assumptions": [
                    "Primitive interface exposes LunarLander position, velocity, attitude, contacts, and action variables."
                ],
                "risk_notes": [
                    "This is a stable initial schema, not final optimal reward."
                ],
                "primitive_interface_only": True,
            },
        }
