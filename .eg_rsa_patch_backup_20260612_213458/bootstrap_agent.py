from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from eg_rsa.llm.json_parser import extract_json_object


class BootstrapAgent:
    """Generate AST-first reward schemas.

    Two input modes are supported:

    1. Primitive-interface mode, kept for backward compatibility:
       primitive_interface -> initial_schema.

    2. Source-aware mode, preferred for V2.1:
       anonymous task/source summary -> primitive_interface + initial_schema.

    In source-aware mode, the runtime environment name is intentionally not shown
    to the LLM. The LLM must infer task variables from task text and anonymized
    source/source-summary information, then generate a schema using only the
    inferred primitive variables.
    """

    DEFAULT_FUNCTIONS = ["abs", "min", "max", "clip", "sqrt", "exp", "tanh"]
    DEFAULT_COMPONENT_TYPES = ["formula_component", "conditional_formula_component", "event_predicate"]
    DEFAULT_SEMANTIC_ROLES = [
        "dense_guidance",
        "stability_quality",
        "terminal_success",
        "safety_constraint",
        "control_cost",
    ]

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

    def generate_bootstrap_from_source(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Infer primitive interface and initial schema from anonymous source input."""
        prompt = self._build_source_prompt(task_spec)
        self.last_prompt = prompt

        if self.llm_client is None:
            primitive_interface = self._primitive_interface_from_task_spec(task_spec)
            result = self._fallback_bootstrap(primitive_interface, primitive_interface.get("task_description", ""))
            result["primitive_interface"] = primitive_interface
            result.setdefault("bootstrap_report", {})["source_aware_bootstrap"] = True
            result["bootstrap_report"]["used_llm"] = False
            self.last_response_text = json.dumps(result, indent=2, ensure_ascii=False)
            return result

        response_text = self.llm_client.generate(prompt)
        self.last_response_text = response_text
        parsed = extract_json_object(response_text)
        return self._normalize_source_bootstrap(parsed, task_spec)

    @staticmethod
    def _ast_grammar() -> Dict[str, Any]:
        return {
            "leaf_nodes": [
                {"var": "<allowed_numeric_variable>"},
                {"var": "<allowed_boolean_variable>"},
                {"const": 0.5},
                {"bool": True},
            ],
            "numeric_ops": [
                {"op": "add", "args": [{"var": "<allowed_numeric_variable_1>"}, {"var": "<allowed_numeric_variable_2>"}]},
                {"op": "sub", "left": {"const": 1.0}, "right": {"var": "<allowed_numeric_variable>"}},
                {"op": "mul", "args": [{"const": 0.5}, {"op": "abs", "arg": {"var": "<allowed_numeric_variable>"}}]},
                {"op": "div", "left": {"var": "<allowed_numeric_variable>"}, "right": {"const": 2.0}},
                {"op": "neg", "arg": {"var": "<allowed_action_variable>"}},
                {"op": "abs", "arg": {"var": "<allowed_numeric_variable>"}},
                {"op": "min", "args": [{"var": "<allowed_numeric_variable>"}, {"const": 1.0}]},
                {"op": "max", "args": [{"var": "<allowed_numeric_variable>"}, {"const": 0.0}]},
                {"op": "clip", "args": [{"var": "<allowed_numeric_variable>"}, {"const": -1.0}, {"const": 1.0}]},
            ],
            "boolean_ops": [
                {"op": "and", "args": [{"var": "<allowed_boolean_variable_1>"}, {"var": "<allowed_boolean_variable_2>"}]},
                {"op": "or", "args": [{"var": "<allowed_boolean_variable_1>"}, {"var": "<allowed_boolean_variable_2>"}]},
                {"op": "not", "arg": {"var": "<allowed_boolean_variable>"}},
                {"op": "lt", "left": {"op": "abs", "arg": {"var": "<allowed_numeric_variable>"}}, "right": {"const": 0.4}},
                {"op": "gt", "left": {"op": "abs", "arg": {"var": "<allowed_numeric_variable>"}}, "right": {"const": 0.6}},
            ],
        }

    @staticmethod
    def _build_prompt(primitive_interface: Dict[str, Any], task_description: str) -> str:
        visible_interface = BootstrapAgent._sanitize_interface_for_llm(primitive_interface)
        allowed_vars = visible_interface.get("allowed_formula_variables", [])
        semantic_roles = visible_interface.get("semantic_roles", [])

        output_shape = BootstrapAgent._reward_output_shape(include_primitive_interface=False)

        return f"""
You are the EG-RSA AST bootstrap agent.

You receive a derived primitive RL task interface. You must output AST-first RewardSchema JSON.
Do not output Python code. Do not output string formulas. Do not output condition strings.

Important input-boundary note:
- The runtime environment name is not part of the reward-generation input.
- Do not guess or mention benchmark/environment names.
- Do not assume task-specific variable names unless they appear in the visible primitive interface.
- The AST grammar and Required output shape below contain placeholders only.
- Every placeholder must be replaced with variables from Allowed variables.
- Do not copy placeholder names into the final JSON.

Task description:
{task_description or visible_interface.get("task_description", "")}

Visible primitive interface:
{json.dumps(visible_interface, indent=2, ensure_ascii=False)}

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
7. Use compact schemas: 3-7 components and 0-3 event rules.
8. Use semantic_role for every component/event.
9. Action cost must be a formula_component with semantic_role="control_cost" and reward-signed negative formula_ast when action/control variables are available.
10. Output JSON only.

Required output shape:
{json.dumps(output_shape, indent=2, ensure_ascii=False)}
""".strip()

    @staticmethod
    def _build_source_prompt(task_spec: Dict[str, Any]) -> str:
        visible_source = BootstrapAgent._sanitize_source_task_for_llm(task_spec)
        output_shape = BootstrapAgent._reward_output_shape(include_primitive_interface=True)
        return f"""
You are the EG-RSA source-aware bootstrap agent.

Your input is an anonymized task description plus anonymized environment source/source-summary information.
Your output must contain both:
1. a derived primitive_interface, and
2. an AST-first initial reward schema that uses only variables from that primitive_interface.

Critical anti-leakage constraints:
1. The benchmark/environment name is intentionally hidden. Do not guess or mention it.
2. Do not use benchmark-specific prior knowledge from a name.
3. Use only the task description and anonymized source/source-summary information below.
4. Do not output Python code. Do not output string formulas. Do not output condition strings.
5. Variable names must be short snake_case and must not include benchmark names.
6. If actions are continuous, use action_mapping.type="continuous_indices".
7. If actions are discrete and action meanings are provided, use action_mapping.type="discrete_lookup".
8. If task completion is not naturally event-based, event_rules may be empty.
9. Output JSON only.

Anonymized task/source input:
{json.dumps(visible_source, indent=2, ensure_ascii=False)}

AST grammar:
{json.dumps(BootstrapAgent._ast_grammar(), indent=2, ensure_ascii=False)}

Required output shape:
{json.dumps(output_shape, indent=2, ensure_ascii=False)}
""".strip()

    @staticmethod
    def _reward_output_shape(include_primitive_interface: bool) -> Dict[str, Any]:
        shape: Dict[str, Any] = {}
        if include_primitive_interface:
            shape["source_understanding"] = {
                "task_objective": "objective inferred from task/source text",
                "observation_semantics": [],
                "action_semantics": [],
                "uncertainties": [],
            }
            shape["primitive_interface"] = {
                "version": 1,
                "input_boundary": "anonymous_source_to_primitive_interface",
                "identity_hidden_from_llm": True,
                "task_description": "copy task objective without environment name",
                "observation_variables": [
                    {"name": "short_snake_case_obs", "description": "meaning", "type": "float_or_bool"}
                ],
                "action_variables": [
                    {"name": "short_snake_case_action", "description": "meaning", "type": "float"}
                ],
                "observation_mapping": {"short_snake_case_obs": 0},
                "action_mapping": {"type": "continuous_indices_or_discrete_lookup", "variables": {"short_snake_case_action": 0}},
                "allowed_formula_variables": ["all observation and action variable names"],
                "allowed_formula_functions": BootstrapAgent.DEFAULT_FUNCTIONS,
                "semantic_roles": BootstrapAgent.DEFAULT_SEMANTIC_ROLES,
            }
        shape.update(
            {
                "reward_blueprint": {
                    "task_objective": "one-sentence objective inferred from task text",
                    "primitive_variable_roles": {
                        "progress_variables": ["names selected only from allowed variables"],
                        "safety_variables": ["names selected only from allowed variables"],
                        "terminal_evidence_variables": ["names selected only from allowed variables"],
                        "control_variables": ["names selected only from allowed action/control variables"],
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
                        "input_boundary": "anonymous_source_conditioned" if include_primitive_interface else "primitive_interface_conditioned",
                    },
                    "components": [
                        {
                            "name": "r_task_progress_example",
                            "type": "formula_component",
                            "weight": 1.0,
                            "formula_ast": {
                                "op": "sub",
                                "left": {"const": 1.0},
                                "right": {"op": "min", "args": [{"op": "abs", "arg": {"var": "<replace_with_allowed_numeric_variable>"}}, {"const": 1.0}]},
                            },
                            "params": {
                                "formula_ast": {
                                    "op": "sub",
                                    "left": {"const": 1.0},
                                    "right": {"op": "min", "args": [{"op": "abs", "arg": {"var": "<replace_with_allowed_numeric_variable>"}}, {"const": 1.0}]},
                                }
                            },
                            "clip": [0.0, 1.0],
                            "enabled": True,
                            "semantic_role": "dense_guidance",
                            "reward_timing": "dense",
                            "behavior_channel": "progress",
                        }
                    ],
                    "event_rules": [],
                },
                "diagnostics": {"expected_failure_modes": [], "risk_notes": []},
                "bootstrap_report": {
                    "design_rationale": "...",
                    "assumptions": [],
                    "risk_notes": [],
                    "identity_hidden_from_llm": include_primitive_interface,
                    "raw_env_code_input": False,
                },
            }
        )
        return shape

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
        schema["metadata"].setdefault("input_boundary", primitive_interface.get("input_boundary", "primitive_interface_conditioned"))
        schema["metadata"].setdefault("raw_env_code_input", bool(primitive_interface.get("raw_env_code_input", False)))
        schema["metadata"].setdefault("identity_hidden_from_llm", bool(primitive_interface.get("identity_hidden_from_llm", False)))
        parsed["initial_schema"] = schema
        parsed.setdefault("diagnostics", {})
        parsed.setdefault("bootstrap_report", {})
        parsed["bootstrap_report"].setdefault("primitive_interface_only", True)
        parsed["bootstrap_report"].setdefault("raw_env_code_input", bool(primitive_interface.get("raw_env_code_input", False)))
        parsed["bootstrap_report"].setdefault("identity_hidden_from_llm", bool(primitive_interface.get("identity_hidden_from_llm", False)))
        return parsed

    @classmethod
    def _normalize_source_bootstrap(cls, parsed: Dict[str, Any], task_spec: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            raise ValueError("Source-aware bootstrap response must parse to a JSON object")
        primitive_interface = parsed.get("primitive_interface")
        if not isinstance(primitive_interface, dict):
            raise ValueError("Source-aware bootstrap response missing primitive_interface")
        primitive_interface = cls._normalize_primitive_interface(primitive_interface, task_spec)
        parsed["primitive_interface"] = primitive_interface
        normalized = cls._normalize(parsed, primitive_interface)
        normalized.setdefault("bootstrap_report", {})["source_aware_bootstrap"] = True
        normalized["bootstrap_report"]["identity_hidden_from_llm"] = True
        normalized["bootstrap_report"]["primitive_interface_generated"] = True
        return normalized

    @classmethod
    def _normalize_primitive_interface(cls, interface: Dict[str, Any], task_spec: Dict[str, Any]) -> Dict[str, Any]:
        obs = cls._normalize_variable_list(interface.get("observation_variables"))
        act = cls._normalize_variable_list(interface.get("action_variables"))
        if not obs:
            raise ValueError("primitive_interface missing observation_variables")
        observation_mapping = interface.get("observation_mapping")
        if not isinstance(observation_mapping, dict) or not observation_mapping:
            observation_mapping = {item["name"]: idx for idx, item in enumerate(obs)}
        action_mapping = interface.get("action_mapping") if isinstance(interface.get("action_mapping"), dict) else {}
        allowed = interface.get("allowed_formula_variables")
        if not isinstance(allowed, list) or not allowed:
            allowed = [item["name"] for item in obs + act]
        return {
            "version": 1,
            "purpose": "Generated primitive task interface inferred inside BootstrapAgent from anonymized task/source input.",
            "input_boundary": "anonymous_source_to_primitive_interface",
            "identity_hidden_from_llm": True,
            "raw_env_code_input": bool(task_spec.get("raw_env_code_input", False)),
            "env_code_parser": "source_aware_bootstrap_agent",
            "task_description": str(task_spec.get("task_description", "") or interface.get("task_description", "")),
            "observation_variables": obs,
            "action_variables": act,
            "allowed_formula_variables": [str(x) for x in allowed],
            "allowed_formula_functions": interface.get("allowed_formula_functions") or task_spec.get("allowed_formula_functions") or cls.DEFAULT_FUNCTIONS,
            "allowed_component_types_v2": task_spec.get("allowed_component_types_v2") or cls.DEFAULT_COMPONENT_TYPES,
            "bootstrap_requirements": interface.get("bootstrap_requirements") or cls._bootstrap_requirements(task_spec),
            "semantic_roles": interface.get("semantic_roles") or task_spec.get("semantic_roles") or cls.DEFAULT_SEMANTIC_ROLES,
            "observation_mapping": {str(k): int(v) for k, v in observation_mapping.items()},
            "bootstrap_interface_policy": {
                "principle": "Reward bootstrap receives a derived interface, not a runtime environment name.",
                "formula_boundary": "Generated formulas must use only allowed_formula_variables and allowed_formula_functions.",
                "not_exposed_to_llm": ["runtime environment name", "gym_id", "benchmark name", "previous reward schema"],
            },
            "action_mapping": action_mapping,
        }

    @staticmethod
    def _normalize_variable_list(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and item.get("name"):
                typ = str(item.get("type", "float")).lower()
                out.append(
                    {
                        "name": str(item["name"]),
                        "description": str(item.get("description", "")),
                        "type": "bool" if typ in {"bool", "boolean"} else "float",
                    }
                )
        return out

    @staticmethod
    def _sanitize_interface_for_llm(primitive_interface: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = {
            "task_description",
            "observation_variables",
            "action_variables",
            "allowed_formula_variables",
            "allowed_formula_functions",
            "allowed_component_types_v2",
            "bootstrap_requirements",
            "semantic_roles",
            "observation_mapping",
            "action_mapping",
            "bootstrap_interface_policy",
        }
        return {k: copy.deepcopy(v) for k, v in (primitive_interface or {}).items() if k in allowed_keys}

    @staticmethod
    def _sanitize_source_task_for_llm(task_spec: Dict[str, Any]) -> Dict[str, Any]:
        policy = task_spec.get("interface_generation_policy", {}) or {}
        terms = list(policy.get("identity_redaction_terms", []) or [])
        blocked_keys = {"env", "env_id", "gym_id", "name", "runtime_env_id", "source_file_hint", "benchmark_name"}

        def redact(value: Any) -> Any:
            if isinstance(value, str):
                out = value
                for term in terms:
                    if term:
                        out = re.sub(re.escape(str(term)), "<REDACTED_ENV_IDENTITY>", out, flags=re.IGNORECASE)
                return out
            if isinstance(value, list):
                return [redact(x) for x in value]
            if isinstance(value, dict):
                return {k: redact(v) for k, v in value.items() if k not in blocked_keys}
            return value

        visible: Dict[str, Any] = {
            "task_description": redact(task_spec.get("task_description", "")),
            "environment_source": redact(task_spec.get("environment_source", {}) or {}),
            "interface_generation_policy": redact(policy),
        }
        for key in ["source_summary", "step_summary", "observation_space_description", "action_space_description"]:
            if key in task_spec:
                visible[key] = redact(task_spec[key])
        return visible

    @staticmethod
    def _bootstrap_requirements(task_spec: Dict[str, Any]) -> Dict[str, Any]:
        req = {
            "must_generate_initial_schema": True,
            "must_generate_diagnostic_predicates": True,
            "must_assign_semantic_roles": True,
            "must_include_hacking_risk_notes": True,
            "official_environment_reward_forbidden_as_feedback": True,
            "posthoc_oracle_evaluation_allowed": True,
            "no_v1_schema_exposure": True,
            "no_v1_diagnostic_metric_exposure": True,
            "no_v1_task_description_file_exposure": True,
            "formula_must_use_only_primitive_variables": True,
        }
        if isinstance(task_spec.get("bootstrap_requirements"), dict):
            req.update(task_spec["bootstrap_requirements"])
        return req

    @staticmethod
    def _split_allowed_variables(primitive_interface: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        allowed = list(primitive_interface.get("allowed_formula_variables", []) or [])
        action_names = {item.get("name") for item in primitive_interface.get("action_variables", []) or [] if isinstance(item, dict) and item.get("name")}
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
        if not numeric:
            numeric = [x for x in allowed if x not in action_names]
        return numeric, boolean, actions

    @classmethod
    def _primitive_interface_from_task_spec(cls, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        obs = cls._normalize_variable_list(task_spec.get("observation_variables") or (task_spec.get("environment_source", {}) or {}).get("observation_variables"))
        act = cls._normalize_variable_list(task_spec.get("action_variables") or (task_spec.get("environment_source", {}) or {}).get("action_variables"))
        if not obs:
            raise ValueError("No observation variables available for fallback source bootstrap")
        observation_mapping = {item["name"]: idx for idx, item in enumerate(obs)}
        allowed = [item["name"] for item in obs + act]
        return {
            "version": 1,
            "input_boundary": "anonymous_source_to_primitive_interface_fallback",
            "identity_hidden_from_llm": True,
            "task_description": task_spec.get("task_description", ""),
            "observation_variables": obs,
            "action_variables": act,
            "allowed_formula_variables": allowed,
            "allowed_formula_functions": task_spec.get("allowed_formula_functions") or cls.DEFAULT_FUNCTIONS,
            "semantic_roles": task_spec.get("semantic_roles") or cls.DEFAULT_SEMANTIC_ROLES,
            "observation_mapping": observation_mapping,
            "action_mapping": task_spec.get("action_mapping") or (task_spec.get("environment_source", {}) or {}).get("action_mapping", {}),
        }

    @staticmethod
    def _fallback_bootstrap(primitive_interface: Dict[str, Any], task_description: str = "") -> Dict[str, Any]:
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
            "phase_structure": [],
            "component_blueprint": [],
            "anti_exploit_principles": [
                "Dense reward should not dominate sparse completion evidence.",
                "Control cost should remain small and bounded.",
                "Fallback schema is not a task-specific optimal reward.",
            ],
        }
        if numeric_vars:
            selected = numeric_vars[:3]
            state_error = add(*[abs_(var(x)) for x in selected]) if len(selected) > 1 else abs_(var(selected[0]))
            progress_ast = sub(const(1.0), min_(state_error, const(1.0)))
            components.append({"name": "r_generic_state_guidance", "type": "formula_component", "weight": 0.5, "formula_ast": progress_ast, "params": {"formula_ast": progress_ast}, "clip": [0.0, 1.0], "enabled": True, "semantic_role": "dense_guidance", "reward_timing": "dense", "behavior_channel": "progress"})
        if action_vars:
            action_mag = add(*[abs_(var(x)) for x in action_vars]) if len(action_vars) > 1 else abs_(var(action_vars[0]))
            control_ast = neg(action_mag)
            components.append({"name": "r_generic_control_cost", "type": "formula_component", "weight": 0.05, "formula_ast": control_ast, "params": {"formula_ast": control_ast}, "clip": [-1.0, 0.0], "enabled": True, "semantic_role": "control_cost", "reward_timing": "dense", "behavior_channel": "action"})
        if bool_vars:
            selected_bool = bool_vars[:2]
            success_ast = and_(*[var(x) for x in selected_bool]) if len(selected_bool) > 1 else var(selected_bool[0])
            event_rules.append({"name": "r_generic_event_evidence", "type": "event_predicate", "weight": 30.0, "condition": {"expr_ast": success_ast, "duration_steps": 1}, "one_time": True, "enabled": True, "semantic_role": "terminal_success", "reward_timing": "sparse_event", "behavior_channel": "completion"})
        if not components:
            zero_ast = {"const": 0.0}
            components.append({"name": "r_zero_placeholder", "type": "formula_component", "weight": 1.0, "formula_ast": zero_ast, "params": {"formula_ast": zero_ast}, "clip": [0.0, 0.0], "enabled": True, "semantic_role": "dense_guidance", "reward_timing": "dense", "behavior_channel": "placeholder"})
        schema = {
            "version": 2,
            "metadata": {
                "source": "fallback_bootstrap_ast",
                "formula_ir": "ast",
                "task": task,
                "reward_blueprint_present": True,
                "input_boundary": primitive_interface.get("input_boundary", "primitive_interface_conditioned"),
                "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
                "identity_hidden_from_llm": bool(primitive_interface.get("identity_hidden_from_llm", False)),
            },
            "components": components,
            "event_rules": event_rules,
        }
        return {
            "reward_blueprint": blueprint,
            "initial_schema": schema,
            "diagnostics": {"expected_failure_modes": ["fallback schema is generic and may be under-specified"], "risk_notes": ["Fallback AST schema is for robustness only."]},
            "bootstrap_report": {
                "design_rationale": "Task-neutral fallback schema built from primitive variables.",
                "assumptions": ["Primitive interface quality determines bootstrap quality."],
                "risk_notes": ["Use LLM bootstrap for serious experiments."],
                "primitive_interface_only": True,
                "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
                "identity_hidden_from_llm": bool(primitive_interface.get("identity_hidden_from_llm", False)),
            },
        }
