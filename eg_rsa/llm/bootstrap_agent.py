from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from eg_rsa.llm.json_parser import extract_json_object


class BootstrapAgent:
    """Generate initial AST-first reward schema from a primitive-only task interface.

    Design boundary:
    - This agent does not read raw env.py / step() directly.
    - It receives a primitive task interface that has already exposed observation
      variables, action variables, safe formula functions, semantic roles, and task text.
    - The prompt template is intentionally task-neutral: examples use placeholders
      rather than hard-coded LunarLander variable names.
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
    def _ast_grammar() -> Dict[str, Any]:
        """Task-neutral AST grammar.

        Placeholders are intentionally not real variables. The prompt instructs the
        LLM to replace them with names from primitive_interface.allowed_formula_variables.
        """
        return {
            "leaf_nodes": [
                {"var": "<allowed_numeric_variable>"},
                {"var": "<allowed_boolean_variable>"},
                {"const": 0.5},
                {"bool": True},
            ],
            "numeric_ops": [
                {
                    "op": "add",
                    "args": [
                        {"var": "<allowed_numeric_variable_1>"},
                        {"var": "<allowed_numeric_variable_2>"},
                    ],
                },
                {
                    "op": "sub",
                    "left": {"const": 1.0},
                    "right": {"var": "<allowed_numeric_variable>"},
                },
                {
                    "op": "mul",
                    "args": [
                        {"const": 0.5},
                        {"op": "abs", "arg": {"var": "<allowed_numeric_variable>"}},
                    ],
                },
                {
                    "op": "div",
                    "left": {"var": "<allowed_numeric_variable>"},
                    "right": {"const": 2.0},
                },
                {
                    "op": "neg",
                    "arg": {"var": "<allowed_action_variable>"},
                },
                {
                    "op": "abs",
                    "arg": {"var": "<allowed_numeric_variable>"},
                },
                {
                    "op": "min",
                    "args": [
                        {"var": "<allowed_numeric_variable>"},
                        {"const": 1.0},
                    ],
                },
                {
                    "op": "max",
                    "args": [
                        {"var": "<allowed_numeric_variable>"},
                        {"const": 0.0},
                    ],
                },
                {
                    "op": "clip",
                    "args": [
                        {"var": "<allowed_numeric_variable>"},
                        {"const": -1.0},
                        {"const": 1.0},
                    ],
                },
            ],
            "boolean_ops": [
                {
                    "op": "and",
                    "args": [
                        {"var": "<allowed_boolean_variable_1>"},
                        {"var": "<allowed_boolean_variable_2>"},
                    ],
                },
                {
                    "op": "or",
                    "args": [
                        {"var": "<allowed_boolean_variable_1>"},
                        {"var": "<allowed_boolean_variable_2>"},
                    ],
                },
                {
                    "op": "not",
                    "arg": {"var": "<allowed_boolean_variable>"},
                },
                {
                    "op": "lt",
                    "left": {"op": "abs", "arg": {"var": "<allowed_numeric_variable>"}},
                    "right": {"const": 0.4},
                },
                {
                    "op": "gt",
                    "left": {"op": "abs", "arg": {"var": "<allowed_numeric_variable>"}},
                    "right": {"const": 0.6},
                },
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
                    "progress_variables": [
                        "names selected only from allowed variables"
                    ],
                    "safety_variables": [
                        "names selected only from allowed variables"
                    ],
                    "terminal_evidence_variables": [
                        "names selected only from allowed variables"
                    ],
                    "control_variables": [
                        "names selected only from allowed action/control variables"
                    ],
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
                    "input_boundary": "primitive_interface_conditioned",
                },
                "components": [
                    {
                        "name": "r_task_progress_example",
                        "type": "formula_component",
                        "weight": 1.0,
                        "formula_ast": {
                            "op": "sub",
                            "left": {"const": 1.0},
                            "right": {
                                "op": "min",
                                "args": [
                                    {
                                        "op": "abs",
                                        "arg": {"var": "<replace_with_allowed_numeric_variable>"},
                                    },
                                    {"const": 1.0},
                                ],
                            },
                        },
                        "params": {
                            "formula_ast": {
                                "op": "sub",
                                "left": {"const": 1.0},
                                "right": {
                                    "op": "min",
                                    "args": [
                                        {
                                            "op": "abs",
                                            "arg": {"var": "<replace_with_allowed_numeric_variable>"},
                                        },
                                        {"const": 1.0},
                                    ],
                                },
                            }
                        },
                        "clip": [0.0, 1.0],
                        "enabled": True,
                        "semantic_role": "dense_guidance",
                        "reward_timing": "dense",
                        "behavior_channel": "progress",
                    }
                ],
                "event_rules": [
                    {
                        "name": "r_task_completion_example",
                        "type": "event_predicate",
                        "weight": 50.0,
                        "condition": {
                            "expr_ast": {
                                "op": "and",
                                "args": [
                                    {"var": "<replace_with_allowed_boolean_or_threshold_predicate>"},
                                    {
                                        "op": "lt",
                                        "left": {
                                            "op": "abs",
                                            "arg": {"var": "<replace_with_allowed_numeric_variable>"},
                                        },
                                        "right": {"const": 0.5},
                                    },
                                ],
                            },
                            "duration_steps": 1,
                        },
                        "one_time": True,
                        "enabled": True,
                        "semantic_role": "terminal_success",
                        "reward_timing": "sparse_event",
                        "behavior_channel": "completion",
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
                "raw_env_code_input": False,
                "eureka_like_input_status": "not_yet_implemented",
            },
        }

        return f"""
You are the EG-RSA-V2 AST bootstrap agent.

You receive a primitive-only RL task interface. You must output AST-first RewardSchema JSON.
Do not output Python code. Do not output string formulas. Do not output condition strings.

Important input-boundary note:
- The current system is primitive-interface-conditioned.
- You are NOT reading raw env.py or step() code in this bootstrap stage.
- Do not assume task-specific variable names unless they appear in the Primitive interface.
- The AST grammar and Required output shape below contain placeholders only.
- Every placeholder must be replaced with variables from Allowed variables.
- Do not copy placeholder names such as <allowed_numeric_variable> into the final JSON.

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
5. All AST variables must come from Allowed variables.
6. Placeholder names in examples are invalid and must not appear in the final JSON.
7. Use compact schemas: 3-7 components and 1-3 event rules.
8. Use semantic_role for every component/event.
9. Action cost must be a formula_component with semantic_role="control_cost" and reward-signed negative formula_ast when action/control variables are available.
10. Output JSON only.

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
        schema["metadata"].setdefault("input_boundary", "primitive_interface_conditioned")
        schema["metadata"].setdefault("raw_env_code_input", False)
        schema["metadata"].setdefault("eureka_like_input_status", "planned_not_current")
        parsed["initial_schema"] = schema
        parsed.setdefault("diagnostics", {})
        parsed.setdefault("bootstrap_report", {})
        parsed["bootstrap_report"].setdefault("primitive_interface_only", True)
        parsed["bootstrap_report"].setdefault("raw_env_code_input", False)
        parsed["bootstrap_report"].setdefault("eureka_like_input_status", "planned_not_current")
        parsed["bootstrap_report"].setdefault("primitive_env", primitive_interface.get("env"))
        return parsed

    @staticmethod
    def _split_allowed_variables(primitive_interface: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        allowed = list(primitive_interface.get("allowed_formula_variables", []) or [])
        action_names = {
            item.get("name")
            for item in primitive_interface.get("action_variables", []) or []
            if isinstance(item, dict) and item.get("name")
        }

        bool_names = set()
        numeric_names = set()

        for item in primitive_interface.get("observation_variables", []) or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            typ = str(item.get("type", "")).lower()
            if typ in {"bool", "boolean"}:
                bool_names.add(name)
            else:
                numeric_names.add(name)

        for item in primitive_interface.get("action_variables", []) or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            typ = str(item.get("type", "")).lower()
            if typ in {"bool", "boolean"}:
                bool_names.add(name)
            else:
                numeric_names.add(name)

        numeric = [x for x in allowed if x in numeric_names or (x not in bool_names and x not in action_names)]
        boolean = [x for x in allowed if x in bool_names]
        actions = [x for x in allowed if x in action_names]

        # Fallback if interface does not provide types.
        if not numeric:
            numeric = [x for x in allowed if x not in action_names]
        return numeric, boolean, actions

    @staticmethod
    def _fallback_bootstrap(
        primitive_interface: Dict[str, Any],
        task_description: str = "",
    ) -> Dict[str, Any]:
        """Task-neutral conservative fallback used only when no LLM client is available.

        This fallback intentionally avoids LunarLander-specific variables. It builds a
        minimal valid schema from whatever variables the primitive interface exposes.
        Real experiments should use an LLM bootstrap or a fixed LLM-generated schema.
        """
        task = task_description or primitive_interface.get("task_description", "")
        numeric_vars, bool_vars, action_vars = BootstrapAgent._split_allowed_variables(primitive_interface)

        def var(name):
            return {"var": name}

        def const(value):
            return {"const": value}

        def abs_(x):
            return {"op": "abs", "arg": x}

        def add(*args):
            return {"op": "add", "args": list(args)}

        def neg(x):
            return {"op": "neg", "arg": x}

        def min_(*args):
            return {"op": "min", "args": list(args)}

        def sub(left, right):
            return {"op": "sub", "left": left, "right": right}

        def and_(*args):
            return {"op": "and", "args": list(args)}

        components: List[Dict[str, Any]] = []
        event_rules: List[Dict[str, Any]] = []

        blueprint = {
            "task_objective": task or "Goal-directed control task",
            "primitive_variable_roles": {
                "progress_variables": numeric_vars[:3],
                "safety_variables": numeric_vars[:5],
                "terminal_evidence_variables": bool_vars[:3],
                "control_variables": action_vars,
            },
            "phase_structure": [
                {
                    "phase": "task_progress",
                    "purpose": "provide conservative dense shaping from primitive state variables",
                    "reward_intent": "avoid unbounded reward and leave room for EG-RSA edits",
                },
                {
                    "phase": "control_regularization",
                    "purpose": "discourage excessive action magnitude when actions are exposed",
                    "reward_intent": "prevent trivial high-control strategies",
                },
                {
                    "phase": "completion_or_event_evidence",
                    "purpose": "use primitive boolean evidence only when provided",
                    "reward_intent": "one-time sparse event evidence without task-specific assumptions",
                },
            ],
            "component_blueprint": [],
            "anti_exploit_principles": [
                "Dense reward should not dominate sparse completion evidence.",
                "Control cost should remain small and bounded.",
                "Fallback schema is not a task-specific optimal reward.",
            ],
        }

        if numeric_vars:
            # Conservative state regularizer: 1 - min(sum(abs(selected_vars)), 1).
            selected = numeric_vars[:3]
            state_error = add(*[abs_(var(x)) for x in selected]) if len(selected) > 1 else abs_(var(selected[0]))
            progress_ast = sub(const(1.0), min_(state_error, const(1.0)))
            components.append(
                {
                    "name": "r_generic_state_guidance",
                    "type": "formula_component",
                    "weight": 0.5,
                    "formula_ast": progress_ast,
                    "params": {"formula_ast": progress_ast},
                    "clip": [0.0, 1.0],
                    "enabled": True,
                    "semantic_role": "dense_guidance",
                    "reward_timing": "dense",
                    "behavior_channel": "progress",
                }
            )
            blueprint["component_blueprint"].append(
                {
                    "name": "r_generic_state_guidance",
                    "role": "dense_guidance",
                    "phase": "task_progress",
                    "variables": selected,
                }
            )

        if action_vars:
            action_mag = add(*[abs_(var(x)) for x in action_vars]) if len(action_vars) > 1 else abs_(var(action_vars[0]))
            control_ast = neg(action_mag)
            components.append(
                {
                    "name": "r_generic_control_cost",
                    "type": "formula_component",
                    "weight": 0.05,
                    "formula_ast": control_ast,
                    "params": {"formula_ast": control_ast},
                    "clip": [-1.0, 0.0],
                    "enabled": True,
                    "semantic_role": "control_cost",
                    "reward_timing": "dense",
                    "behavior_channel": "action",
                }
            )
            blueprint["component_blueprint"].append(
                {
                    "name": "r_generic_control_cost",
                    "role": "control_cost",
                    "phase": "control_regularization",
                    "variables": action_vars,
                }
            )

        if len(bool_vars) >= 1:
            selected_bool = bool_vars[:2]
            success_ast = and_(*[var(x) for x in selected_bool]) if len(selected_bool) > 1 else var(selected_bool[0])
            event_rules.append(
                {
                    "name": "r_generic_event_evidence",
                    "type": "event_predicate",
                    "weight": 30.0,
                    "condition": {"expr_ast": success_ast, "duration_steps": 1},
                    "one_time": True,
                    "enabled": True,
                    "semantic_role": "terminal_success",
                    "reward_timing": "sparse_event",
                    "behavior_channel": "completion",
                }
            )
            blueprint["component_blueprint"].append(
                {
                    "name": "r_generic_event_evidence",
                    "role": "terminal_success",
                    "phase": "completion_or_event_evidence",
                    "variables": selected_bool,
                }
            )

        if not components:
            # Last-resort constant zero component keeps schema shape valid without
            # pretending to solve the task.
            zero_ast = {"const": 0.0}
            components.append(
                {
                    "name": "r_zero_placeholder",
                    "type": "formula_component",
                    "weight": 1.0,
                    "formula_ast": zero_ast,
                    "params": {"formula_ast": zero_ast},
                    "clip": [0.0, 0.0],
                    "enabled": True,
                    "semantic_role": "dense_guidance",
                    "reward_timing": "dense",
                    "behavior_channel": "placeholder",
                }
            )

        schema = {
            "version": 2,
            "metadata": {
                "source": "fallback_bootstrap_ast",
                "formula_ir": "ast",
                "task": task,
                "reward_blueprint_present": True,
                "input_boundary": "primitive_interface_conditioned",
                "raw_env_code_input": False,
                "eureka_like_input_status": "planned_not_current",
            },
            "components": components,
            "event_rules": event_rules,
        }

        return {
            "reward_blueprint": blueprint,
            "initial_schema": schema,
            "diagnostics": {
                "expected_failure_modes": [
                    "fallback schema is generic and may be under-specified",
                    "task-specific improvement should be performed by EG-RSA iterations",
                ],
                "risk_notes": [
                    "Fallback AST schema is for robustness only; it is not a final reward design."
                ],
            },
            "bootstrap_report": {
                "design_rationale": "Task-neutral fallback schema built only from variables exposed by primitive_interface.",
                "assumptions": [
                    "No raw env.py or step() parsing is performed in this bootstrap stage.",
                    "Primitive interface quality determines the useful information available to bootstrap.",
                ],
                "risk_notes": [
                    "For serious experiments, use LLM bootstrap or a fixed LLM-generated schema."
                ],
                "primitive_interface_only": True,
                "raw_env_code_input": False,
                "eureka_like_input_status": "planned_not_current",
            },
        }
