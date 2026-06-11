from __future__ import annotations

import json
from typing import Any, Dict, Optional

from eg_rsa.llm.json_parser import extract_json_object


class BootstrapAgent:
    """Generate an initial reward schema from a primitive-only task interface."""

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

        return f"""
You are the EG-RSA-V2 bootstrap agent.

You receive a primitive-only reinforcement-learning task interface.
You are not given any previous reward schema, previous diagnostic metric names,
previous event predicate names, or official environment reward feedback.

Task description:
{task_description or primitive_interface.get("task_description", "")}

Primitive interface:
{json.dumps(primitive_interface, indent=2, ensure_ascii=False)}

Rules:
1. Output JSON only. Do not output markdown.
2. Do not use official environment reward as feedback.
3. Do not assume any hidden task metrics or event names.
4. Reward formulas may use only these variables: {allowed_vars}
5. Reward formulas may call only these functions: {allowed_funcs}
6. Use only these schema item types: {allowed_component_types}
7. Every component and event rule must have semantic_role from: {semantic_roles}
8. You may use ordinary English concept names such as stability, soft landing, or safe descent in item names, but every formula must be expressed only with primitive variables.

Required JSON format:
{{
  "initial_schema": {{
    "version": 2,
    "metadata": {{
      "source": "llm_bootstrap",
      "task": "..."
    }},
    "components": [
      {{
        "name": "r_centering",
        "type": "formula_component",
        "weight": 1.0,
        "formula": "1.0 - min(abs(x), 1.0)",
        "clip": [0.0, 1.0],
        "enabled": true,
        "semantic_role": "dense_guidance",
        "reward_timing": "dense",
        "behavior_channel": "position"
      }}
    ],
    "event_rules": [
      {{
        "name": "r_soft_landing_contact_once",
        "type": "event_predicate",
        "weight": 80.0,
        "condition": {{
          "expression": "left_contact and right_contact and abs(vy) < 0.3 and abs(angle) < 0.3",
          "duration_steps": 3
        }},
        "one_time": true,
        "enabled": true,
        "semantic_role": "terminal_success",
        "reward_timing": "sparse_event",
        "behavior_channel": "success"
      }}
    ]
  }},
  "diagnostics": {{
    "expected_failure_modes": [],
    "risk_notes": []
  }},
  "bootstrap_report": {{
    "design_rationale": "...",
    "assumptions": [],
    "risk_notes": [],
    "primitive_interface_only": true
  }}
}}

Generate a compact schema with 3-6 components and 1-3 event rules.
""".strip()

    @staticmethod
    def _normalize(parsed: Dict[str, Any], primitive_interface: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            raise ValueError("Bootstrap LLM response must parse to a JSON object")

        if "initial_schema" not in parsed:
            raise ValueError("Bootstrap response missing initial_schema")

        schema = dict(parsed["initial_schema"] or {})
        schema.setdefault("version", 2)
        schema.setdefault("metadata", {})
        schema.setdefault("components", [])
        schema.setdefault("event_rules", [])

        for component in schema.get("components", []):
            if isinstance(component, dict):
                BootstrapAgent._normalize_component(component)

        for rule in schema.get("event_rules", []):
            if isinstance(rule, dict):
                BootstrapAgent._normalize_event_rule(rule)

        parsed["initial_schema"] = schema
        parsed.setdefault("diagnostics", {})
        parsed.setdefault("bootstrap_report", {})
        parsed["bootstrap_report"].setdefault("primitive_interface_only", True)
        parsed["bootstrap_report"].setdefault("primitive_env", primitive_interface.get("env"))
        return parsed

    @staticmethod
    def _normalize_component(component: Dict[str, Any]) -> None:
        ctype = component.get("type")
        component.setdefault("enabled", True)
        component.setdefault("weight", 1.0)
        component.setdefault("params", {})

        if "formula" in component:
            component["params"]["formula"] = component["formula"]
        elif "formula" in component.get("params", {}):
            component["formula"] = component["params"]["formula"]

        if ctype == "conditional_formula_component":
            if "condition" in component:
                component["params"]["condition"] = component["condition"]
            elif "condition" in component.get("params", {}):
                component["condition"] = component["params"]["condition"]

    @staticmethod
    def _normalize_event_rule(rule: Dict[str, Any]) -> None:
        rule.setdefault("enabled", True)
        rule.setdefault("weight", 1.0)
        rule.setdefault("one_time", True)

        condition = rule.get("condition", {})
        if isinstance(condition, str):
            rule["condition"] = {"expression": condition, "duration_steps": int(rule.get("duration_steps", 1) or 1)}
        elif isinstance(condition, dict):
            condition.setdefault("duration_steps", int(rule.get("duration_steps", condition.get("duration_steps", 1)) or 1))
            rule["condition"] = condition

    @staticmethod
    def _fallback_bootstrap(
        primitive_interface: Dict[str, Any],
        task_description: str = "",
    ) -> Dict[str, Any]:
        schema = {
            "version": 2,
            "metadata": {
                "source": "fallback_bootstrap",
                "task": task_description or primitive_interface.get("task_description", ""),
            },
            "components": [
                {
                    "name": "r_centering_from_x",
                    "type": "formula_component",
                    "weight": 0.8,
                    "formula": "1.0 - min(abs(x), 1.0)",
                    "params": {"formula": "1.0 - min(abs(x), 1.0)"},
                    "clip": [0.0, 1.0],
                    "enabled": True,
                    "semantic_role": "dense_guidance",
                    "reward_timing": "dense",
                    "behavior_channel": "position",
                },
                {
                    "name": "r_smooth_velocity",
                    "type": "formula_component",
                    "weight": 0.5,
                    "formula": "1.0 - min(abs(vx) + abs(vy), 1.0)",
                    "params": {"formula": "1.0 - min(abs(vx) + abs(vy), 1.0)"},
                    "clip": [0.0, 1.0],
                    "enabled": True,
                    "semantic_role": "stability_quality",
                    "reward_timing": "dense",
                    "behavior_channel": "velocity",
                },
                {
                    "name": "r_upright_attitude",
                    "type": "formula_component",
                    "weight": 0.5,
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
                    "name": "r_soft_two_leg_contact_once",
                    "type": "event_predicate",
                    "weight": 80.0,
                    "condition": {
                        "expression": "left_contact and right_contact and abs(vy) < 0.3 and abs(angle) < 0.3",
                        "duration_steps": 2,
                    },
                    "one_time": True,
                    "enabled": True,
                    "semantic_role": "terminal_success",
                    "reward_timing": "sparse_event",
                    "behavior_channel": "success",
                }
            ],
        }

        return {
            "initial_schema": schema,
            "diagnostics": {
                "expected_failure_modes": [
                    "dense_reward_without_terminal_event",
                    "excessive_action_cost_avoidance",
                    "unstable_contact_behavior",
                ],
                "risk_notes": [
                    "Dense rewards may be exploited without achieving contact.",
                    "Terminal event is one-time to reduce repeated contact exploitation.",
                ],
            },
            "bootstrap_report": {
                "design_rationale": "Fallback bootstrap uses only primitive variables for centering, smooth velocity, upright attitude, control cost, and soft two-leg contact.",
                "assumptions": [
                    "The primitive interface exposes x, vx, vy, angle, angular_velocity, and contact flags."
                ],
                "risk_notes": [
                    "This fallback is conservative and should be replaced by an LLM-generated bootstrap when available."
                ],
                "primitive_interface_only": True,
            },
        }
