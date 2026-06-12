from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class SourceAwareSafeScaffold:
    """Build a deterministic executable reward scaffold from an inferred interface.

    This is not an old hand-written environment reward. It is an environment-name-
    agnostic safety scaffold generated from:
      - task_description in the source-aware task input,
      - primitive_interface inferred by BootstrapAgent,
      - optional reward_blueprint produced by the LLM.

    It is used only when the LLM-produced executable AST schema is invalid. The
    goal is to preserve the source-aware input boundary while guaranteeing that
    the training loop can start from a valid, auditable reward schema.
    """

    @classmethod
    def build(
        cls,
        primitive_interface: Dict[str, Any],
        reward_blueprint: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        reward_blueprint = reward_blueprint or {}
        validation_errors = validation_errors or []
        task_text = str(primitive_interface.get("task_description", "") or reward_blueprint.get("task_objective", ""))
        task_mode = cls._infer_task_mode(task_text)

        numeric_obs, bool_obs, action_vars = cls._split_variables(primitive_interface)
        roles = cls._classify_variables(task_text, numeric_obs, bool_obs, action_vars)

        components: List[Dict[str, Any]] = []
        event_rules: List[Dict[str, Any]] = []

        if task_mode == "locomotion":
            components.extend(cls._locomotion_components(roles))
        else:
            components.extend(cls._landing_or_goal_components(roles))

        components.extend(cls._generic_stability_components(roles))
        components.extend(cls._control_cost_components(action_vars))

        if task_mode == "landing":
            maybe_event = cls._landing_success_event(roles)
            if maybe_event is not None:
                event_rules.append(maybe_event)

        if not components:
            components.append(cls._zero_placeholder())

        blueprint = cls._build_blueprint(task_text, roles, action_vars, task_mode, reward_blueprint)
        schema = {
            "version": 2,
            "metadata": {
                "source": "source_aware_deterministic_safe_scaffold",
                "formula_ir": "ast",
                "input_boundary": primitive_interface.get("input_boundary", "anonymous_source_to_primitive_interface"),
                "identity_hidden_from_llm": bool(primitive_interface.get("identity_hidden_from_llm", True)),
                "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
                "schema_source": "programmatic_scaffold_from_inferred_interface",
                "llm_schema_replaced": True,
                "llm_validation_errors": list(validation_errors),
            },
            "components": components,
            "event_rules": event_rules,
        }
        return {
            "primitive_interface": primitive_interface,
            "reward_blueprint": blueprint,
            "initial_schema": schema,
            "diagnostics": {
                "expected_failure_modes": [
                    "scaffold may be conservative because it avoids environment-name priors",
                    "progress direction may be under-specified if source summary omits progress semantics",
                ],
                "risk_notes": [
                    "LLM-produced source-aware schema failed validation; using deterministic scaffold from inferred interface.",
                    "No previous environment-specific reward schema was used.",
                ],
            },
            "bootstrap_report": {
                "source_aware_bootstrap": True,
                "primitive_interface_generated": True,
                "schema_source": "source_aware_deterministic_safe_scaffold",
                "llm_schema_replaced": True,
                "llm_validation_errors": list(validation_errors),
                "design_rationale": (
                    "The executable schema is generated programmatically from the inferred primitive interface "
                    "to avoid invalid LLM AST structures while preserving anonymous source-aware input."
                ),
                "assumptions": [
                    "Variable roles are inferred from variable names/descriptions and task text, not from environment names.",
                    "Sparse terminal events are optional and are only emitted when executable contact evidence is available.",
                ],
                "risk_notes": [
                    "Safe scaffold is intended as a valid starting point, not a claim of optimal reward design."
                ],
            },
        }

    @staticmethod
    def _infer_task_mode(task_text: str) -> str:
        text = task_text.lower()
        if any(k in text for k in ["land", "landing", "descend", "touchdown", "target zone", "ground contact"]):
            return "landing"
        if any(k in text for k in ["walk", "run", "move forward", "forward", "locomotion", "two-legged", "biped"]):
            return "locomotion"
        return "goal_control"

    @staticmethod
    def _split_variables(primitive_interface: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        numeric_obs: List[Dict[str, Any]] = []
        bool_obs: List[Dict[str, Any]] = []
        for item in primitive_interface.get("observation_variables", []) or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            typ = str(item.get("type", "float")).lower()
            entry = {"name": str(item["name"]), "description": str(item.get("description", "")), "type": "bool" if typ in {"bool", "boolean"} else "float"}
            if entry["type"] == "bool":
                bool_obs.append(entry)
            else:
                numeric_obs.append(entry)
        action_vars: List[Dict[str, Any]] = []
        for item in primitive_interface.get("action_variables", []) or []:
            if isinstance(item, dict) and item.get("name"):
                action_vars.append({"name": str(item["name"]), "description": str(item.get("description", "")), "type": str(item.get("type", "float"))})
        return numeric_obs, bool_obs, action_vars

    @classmethod
    def _classify_variables(
        cls,
        task_text: str,
        numeric_obs: List[Dict[str, Any]],
        bool_obs: List[Dict[str, Any]],
        action_vars: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        roles: Dict[str, List[Dict[str, Any]]] = {
            "center_position": [],
            "height_or_vertical_position": [],
            "forward_progress": [],
            "velocity": [],
            "descent_velocity": [],
            "stability_angle": [],
            "angular_velocity": [],
            "contact": [],
            "action": list(action_vars),
            "generic_numeric": list(numeric_obs),
        }
        for item in numeric_obs:
            text = cls._var_text(item)
            if any(k in text for k in ["horizontal position", "target center", "center", "lateral", "x position", "horiz_pos", "horizontal_pos"]):
                roles["center_position"].append(item)
            if any(k in text for k in ["vertical position", "height", "altitude", "y position", "vert_pos", "vertical_pos"]):
                roles["height_or_vertical_position"].append(item)
            if any(k in text for k in ["forward", "progress", "distance", "horizontal speed", "forward speed", "x velocity", "horiz_vel", "horizontal_vel"]):
                roles["forward_progress"].append(item)
            if any(k in text for k in ["velocity", "speed", "vel"]):
                roles["velocity"].append(item)
            if any(k in text for k in ["vertical velocity", "descent", "descend", "vy", "vert_vel", "vertical_vel"]):
                roles["descent_velocity"].append(item)
            if any(k in text for k in ["angle", "tilt", "upright", "body angle", "hull angle"]):
                roles["stability_angle"].append(item)
            if any(k in text for k in ["angular velocity", "ang_vel", "angular_vel"]):
                roles["angular_velocity"].append(item)
        for item in bool_obs:
            text = cls._var_text(item)
            if any(k in text for k in ["contact", "touch", "ground", "leg"]):
                roles["contact"].append(item)
        return roles

    @staticmethod
    def _var_text(item: Dict[str, Any]) -> str:
        return f"{item.get('name', '')} {item.get('description', '')}".lower().replace("_", " ")

    @classmethod
    def _landing_or_goal_components(cls, roles: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        comps: List[Dict[str, Any]] = []
        if roles["center_position"]:
            ast = cls._one_minus_abs_sum(roles["center_position"][:1])
            comps.append(cls._component("r_scaffold_centering", ast, 0.6, [0.0, 1.0], "dense_guidance", "progress"))
        if roles["velocity"] or roles["descent_velocity"]:
            selected = cls._unique_vars((roles["descent_velocity"] or []) + roles["velocity"])
            ast = cls._one_minus_abs_sum(selected[:2])
            comps.append(cls._component("r_scaffold_gentle_motion", ast, 0.5, [0.0, 1.0], "stability_quality", "stability"))
        if roles["height_or_vertical_position"] and not comps:
            ast = cls._one_minus_abs_sum(roles["height_or_vertical_position"][:1])
            comps.append(cls._component("r_scaffold_goal_position", ast, 0.3, [0.0, 1.0], "dense_guidance", "progress"))
        return comps

    @classmethod
    def _locomotion_components(cls, roles: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        comps: List[Dict[str, Any]] = []
        if roles["forward_progress"]:
            var = roles["forward_progress"][0]["name"]
            ast = {"op": "clip", "args": [{"var": var}, {"const": 0.0}, {"const": 1.0}]}
            comps.append(cls._component("r_scaffold_forward_progress", ast, 0.8, [0.0, 1.0], "dense_guidance", "progress"))
        return comps

    @classmethod
    def _generic_stability_components(cls, roles: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        comps: List[Dict[str, Any]] = []
        selected = cls._unique_vars(roles["stability_angle"][:1] + roles["angular_velocity"][:1])
        if selected:
            ast = cls._one_minus_abs_sum(selected)
            comps.append(cls._component("r_scaffold_upright_stability", ast, 0.4, [0.0, 1.0], "stability_quality", "stability"))
        return comps

    @classmethod
    def _control_cost_components(cls, action_vars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not action_vars:
            return []
        terms = [{"op": "abs", "arg": {"var": item["name"]}} for item in action_vars[:4]]
        mag = terms[0] if len(terms) == 1 else {"op": "add", "args": terms}
        ast = {"op": "neg", "arg": mag}
        return [cls._component("r_scaffold_control_cost", ast, 0.05, [-1.0, 0.0], "control_cost", "control")]

    @classmethod
    def _landing_success_event(cls, roles: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        contacts = roles["contact"][:2]
        if not contacts:
            return None
        args: List[Dict[str, Any]] = [{"var": item["name"]} for item in contacts]
        if roles["center_position"]:
            args.append({"op": "lt", "left": {"op": "abs", "arg": {"var": roles["center_position"][0]["name"]}}, "right": {"const": 0.4}})
        if roles["velocity"] or roles["descent_velocity"]:
            selected = cls._unique_vars((roles["descent_velocity"] or []) + roles["velocity"])
            for item in selected[:2]:
                args.append({"op": "lt", "left": {"op": "abs", "arg": {"var": item["name"]}}, "right": {"const": 0.4}})
        if not args:
            return None
        cond = args[0] if len(args) == 1 else {"op": "and", "args": args}
        return {
            "name": "r_scaffold_contact_success_event",
            "type": "event_predicate",
            "weight": 20.0,
            "condition": {"expr_ast": cond, "duration_steps": 1},
            "one_time": True,
            "enabled": True,
            "semantic_role": "terminal_success",
            "reward_timing": "sparse_event",
            "behavior_channel": "completion",
        }

    @classmethod
    def _one_minus_abs_sum(cls, variables: List[Dict[str, Any]]) -> Dict[str, Any]:
        terms = [{"op": "abs", "arg": {"var": item["name"]}} for item in variables if item.get("name")]
        if not terms:
            return {"const": 0.0}
        total = terms[0] if len(terms) == 1 else {"op": "add", "args": terms}
        return {"op": "sub", "left": {"const": 1.0}, "right": {"op": "min", "args": [total, {"const": 1.0}]}}

    @staticmethod
    def _component(name: str, ast: Dict[str, Any], weight: float, clip: List[float], role: str, channel: str) -> Dict[str, Any]:
        return {
            "name": name,
            "type": "formula_component",
            "weight": float(weight),
            "formula_ast": ast,
            "params": {"formula_ast": ast},
            "clip": clip,
            "enabled": True,
            "semantic_role": role,
            "reward_timing": "dense",
            "behavior_channel": channel,
        }

    @staticmethod
    def _zero_placeholder() -> Dict[str, Any]:
        ast = {"const": 0.0}
        return {
            "name": "r_scaffold_zero_placeholder",
            "type": "formula_component",
            "weight": 1.0,
            "formula_ast": ast,
            "params": {"formula_ast": ast},
            "clip": [0.0, 0.0],
            "enabled": True,
            "semantic_role": "dense_guidance",
            "reward_timing": "dense",
            "behavior_channel": "placeholder",
        }

    @staticmethod
    def _unique_vars(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for item in items:
            name = item.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(item)
        return out

    @staticmethod
    def _build_blueprint(
        task_text: str,
        roles: Dict[str, List[Dict[str, Any]]],
        action_vars: List[Dict[str, Any]],
        task_mode: str,
        original_blueprint: Dict[str, Any],
    ) -> Dict[str, Any]:
        def names(key: str) -> List[str]:
            return [x["name"] for x in roles.get(key, []) if x.get("name")]

        return {
            "task_objective": task_text or original_blueprint.get("task_objective", "anonymous source-aware control task"),
            "source": "source_aware_deterministic_safe_scaffold",
            "task_mode": task_mode,
            "primitive_variable_roles": {
                "progress_variables": names("center_position") + names("forward_progress") + names("height_or_vertical_position")[:1],
                "safety_variables": names("stability_angle") + names("angular_velocity") + names("velocity"),
                "terminal_evidence_variables": names("contact"),
                "control_variables": [x["name"] for x in action_vars if x.get("name")],
            },
            "phase_structure": original_blueprint.get("phase_structure", []),
            "component_blueprint": [],
            "anti_exploit_principles": [
                "Do not use official environment reward as feedback.",
                "Use only inferred primitive variables.",
                "Keep initial dense shaping bounded and auditable.",
            ],
        }
