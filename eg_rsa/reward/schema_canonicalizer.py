from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

from eg_rsa.reward.safe_formula_eval import safe_eval_formula


class SchemaCanonicalizer:
    """Canonicalize raw LLM reward schema before validation/runtime.

    Design principle:
      LLM raw JSON is not the source of truth.
      Canonical schema is the source of truth.

    Responsibilities:
      - normalize role -> semantic_role
      - normalize formula / condition into params
      - normalize event expression into condition.expression
      - normalize event reward -> weight
      - convert formula-bearing action_penalty into formula_component
      - reward-sign control_cost formulas so sampled contribution is <= 0
    """

    DEFAULT_ALLOWED_VARS = {
        "x", "y", "vx", "vy", "angle", "angular_velocity",
        "left_contact", "right_contact", "main_engine", "side_engine",
        "contact", "both_contact",
    }

    @classmethod
    def canonicalize_bootstrap_result(
        cls,
        bootstrap_result: Dict[str, Any],
        primitive_interface: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        result = copy.deepcopy(bootstrap_result)
        schema = result.get("initial_schema")
        blueprint = result.get("reward_blueprint", {}) or {}

        if not isinstance(schema, dict):
            return result, {
                "ok": False,
                "notes": [],
                "errors": ["bootstrap_result.initial_schema is not a dict"],
            }

        canonical_schema, report = cls.canonicalize_schema(
            schema=schema,
            primitive_interface=primitive_interface,
            reward_blueprint=blueprint,
        )
        result["initial_schema"] = canonical_schema
        result.setdefault("bootstrap_report", {})
        result["bootstrap_report"].setdefault("canonicalization_notes", [])
        result["bootstrap_report"]["canonicalization_notes"].extend(report.get("notes", []))
        return result, report

    @classmethod
    def canonicalize_schema(
        cls,
        schema: Dict[str, Any],
        primitive_interface: Dict[str, Any] | None = None,
        reward_blueprint: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        primitive_interface = primitive_interface or {}
        reward_blueprint = reward_blueprint or {}

        data = copy.deepcopy(schema)
        data.setdefault("version", 2)
        data.setdefault("metadata", {})
        data.setdefault("components", [])
        data.setdefault("event_rules", [])

        # Preserve primitive action semantics inside canonical schema metadata
        # so runtime artifacts can reconstruct formula variables consistently.
        if primitive_interface.get("action_mapping"):
            data["metadata"]["action_mapping"] = primitive_interface.get("action_mapping")
        if primitive_interface.get("action_variables"):
            data["metadata"]["action_variables"] = primitive_interface.get("action_variables")
        if primitive_interface.get("allowed_formula_variables"):
            data["metadata"]["allowed_formula_variables"] = primitive_interface.get("allowed_formula_variables")
        if primitive_interface.get("allowed_formula_functions"):
            data["metadata"]["allowed_formula_functions"] = primitive_interface.get("allowed_formula_functions")

        notes: List[str] = []
        errors: List[str] = []

        role_hints = cls._build_blueprint_role_hints(reward_blueprint)

        event_expr_by_name: Dict[str, str] = {}
        canonical_events: List[Dict[str, Any]] = []
        for raw_rule in data.get("event_rules", []) or []:
            if not isinstance(raw_rule, dict):
                errors.append("Non-dict event_rule dropped during canonicalization")
                continue
            rule = cls._canonicalize_event_rule(raw_rule, role_hints, notes)
            expr = cls._event_expr(rule)
            if rule.get("name") and expr:
                event_expr_by_name[str(rule["name"])] = expr
            canonical_events.append(rule)

        canonical_components: List[Dict[str, Any]] = []
        for raw_component in data.get("components", []) or []:
            if not isinstance(raw_component, dict):
                errors.append("Non-dict component dropped during canonicalization")
                continue
            comp = cls._canonicalize_component(
                raw_component,
                role_hints=role_hints,
                event_expr_by_name=event_expr_by_name,
                primitive_interface=primitive_interface,
                notes=notes,
            )
            canonical_components.append(comp)

        data["components"] = canonical_components
        data["event_rules"] = canonical_events

        if notes:
            data["metadata"].setdefault("canonicalization_notes", [])
            data["metadata"]["canonicalization_notes"].extend(notes)

        report = {
            "ok": len(errors) == 0,
            "notes": notes,
            "errors": errors,
            "component_count": len(canonical_components),
            "event_rule_count": len(canonical_events),
        }
        return data, report

    @classmethod
    def _canonicalize_component(
        cls,
        component: Dict[str, Any],
        role_hints: Dict[str, Dict[str, Any]],
        event_expr_by_name: Dict[str, str],
        primitive_interface: Dict[str, Any],
        notes: List[str],
    ) -> Dict[str, Any]:
        comp = copy.deepcopy(component)
        name = str(comp.get("name", ""))

        comp.setdefault("enabled", True)
        comp.setdefault("weight", 1.0)
        comp.setdefault("params", {})

        params = dict(comp.get("params", {}) or {})

        # role -> semantic_role
        if not comp.get("semantic_role") and comp.get("role"):
            comp["semantic_role"] = comp.get("role")
            notes.append(f"Component {name}: copied role -> semantic_role.")

        # blueprint role hints
        hint = cls._match_hint(name, role_hints)
        if hint and not comp.get("semantic_role") and hint.get("role"):
            comp["semantic_role"] = hint["role"]
            notes.append(f"Component {name}: filled semantic_role from blueprint.")
        if hint and not comp.get("reward_timing"):
            comp["reward_timing"] = cls._timing_from_role_phase(hint.get("role"), hint.get("phase"))
        if hint and not comp.get("behavior_channel"):
            comp["behavior_channel"] = cls._channel_from_role_phase(hint.get("role"), hint.get("phase"))

        # formula / condition into params
        if "formula" in comp:
            params["formula"] = comp["formula"]
        elif "formula" in params:
            comp["formula"] = params["formula"]

        if "condition" in comp:
            params["condition"] = comp["condition"]
        elif "condition" in params:
            comp["condition"] = params["condition"]

        ctype = comp.get("type")

        # If LLM emits action_penalty with custom formula, convert to formula_component.
        if ctype == "action_penalty" and params.get("formula"):
            comp["type"] = "formula_component"
            comp.setdefault("semantic_role", "control_cost")
            comp.setdefault("reward_timing", "dense")
            comp.setdefault("behavior_channel", "control")
            ctype = "formula_component"
            notes.append(
                f"Component {name}: converted formula-bearing action_penalty "
                "to formula_component."
            )

        # Expand component condition that references an event-rule name.
        if ctype == "conditional_formula_component":
            cond = params.get("condition")
            if isinstance(cond, str) and cond.strip() in event_expr_by_name:
                expanded = event_expr_by_name[cond.strip()]
                params["condition"] = expanded
                comp["condition"] = expanded
                notes.append(
                    f"Component {name}: expanded condition event alias "
                    f"{cond.strip()!r} into primitive expression."
                )

        # Reward-sign control cost.
        role = comp.get("semantic_role") or params.get("semantic_role")
        if str(role) == "control_cost" and params.get("formula"):
            formula_before = str(params["formula"])
            weight_before = float(comp.get("weight", 1.0) or 1.0)
            formula_after, weight_after, changed = cls._reward_sign_control_cost(
                formula=formula_before,
                weight=weight_before,
                primitive_interface=primitive_interface,
            )
            if changed:
                params["formula"] = formula_after
                comp["formula"] = formula_after
                comp["weight"] = weight_after
                notes.append(
                    f"Component {name}: reward-signed control_cost formula "
                    f"from {formula_before!r} to {formula_after!r}."
                )

        comp["params"] = params

        # Keep top-level formula/condition synchronized for readability.
        if comp.get("type") in {"formula_component", "conditional_formula_component"} and "formula" in params:
            comp["formula"] = params["formula"]
        if comp.get("type") == "conditional_formula_component" and "condition" in params:
            comp["condition"] = params["condition"]

        return comp

    @classmethod
    def _canonicalize_event_rule(
        cls,
        rule: Dict[str, Any],
        role_hints: Dict[str, Dict[str, Any]],
        notes: List[str],
    ) -> Dict[str, Any]:
        out = copy.deepcopy(rule)
        name = str(out.get("name", ""))

        out.setdefault("enabled", True)
        out.setdefault("one_time", True)
        out.setdefault("type", "event_predicate")

        # reward -> weight. Runtime only uses weight, so reward must not be ignored.
        if "reward" in out:
            try:
                reward_value = float(out.get("reward"))
                old_weight = out.get("weight", None)
                out["weight"] = reward_value
                notes.append(
                    f"Event rule {name}: moved reward={reward_value} into weight "
                    f"(old weight={old_weight})."
                )
            except Exception:
                notes.append(f"Event rule {name}: ignored non-numeric reward field.")
            out.pop("reward", None)
        else:
            out.setdefault("weight", 1.0)

        # role -> semantic_role
        if not out.get("semantic_role") and out.get("role"):
            out["semantic_role"] = out.get("role")
            notes.append(f"Event rule {name}: copied role -> semantic_role.")

        hint = cls._match_hint(name, role_hints)
        if hint and not out.get("semantic_role") and hint.get("role"):
            out["semantic_role"] = hint["role"]
            notes.append(f"Event rule {name}: filled semantic_role from blueprint.")

        # Generic fallback: positive one-time event -> terminal_success;
        # negative one-time event -> safety_constraint.
        if not out.get("semantic_role") and bool(out.get("one_time", False)):
            try:
                w = float(out.get("weight", 1.0) or 1.0)
                out["semantic_role"] = "terminal_success" if w >= 0.0 else "safety_constraint"
                notes.append(
                    f"Event rule {name}: inferred semantic_role={out['semantic_role']} "
                    "from one_time sign."
                )
            except Exception:
                pass

        out.setdefault("reward_timing", "sparse_event")
        if not out.get("behavior_channel"):
            role = out.get("semantic_role")
            if role == "terminal_success":
                out["behavior_channel"] = "completion"
            elif role == "safety_constraint":
                out["behavior_channel"] = "safety"

        condition = out.get("condition", {})
        if isinstance(condition, str):
            condition = {"expression": condition}
        elif not isinstance(condition, dict):
            condition = {}
        else:
            condition = dict(condition)

        top_expr = out.get("expression") or out.get("formula")
        if top_expr and not (condition.get("expression") or condition.get("formula")):
            condition["expression"] = str(top_expr)
            notes.append(
                f"Event rule {name}: moved top-level expression/formula "
                "into condition.expression."
            )

        condition.setdefault("duration_steps", int(out.get("duration_steps", condition.get("duration_steps", 1)) or 1))
        out["condition"] = condition

        out.pop("expression", None)
        out.pop("formula", None)
        out.pop("duration_steps", None)
        return out

    @staticmethod
    def _event_expr(rule: Dict[str, Any]) -> str | None:
        condition = rule.get("condition", {})
        if isinstance(condition, dict):
            expr = condition.get("expression") or condition.get("formula")
            return str(expr) if expr else None
        if isinstance(condition, str):
            return condition
        return None

    @classmethod
    def _reward_sign_control_cost(
        cls,
        formula: str,
        weight: float,
        primitive_interface: Dict[str, Any],
    ) -> Tuple[str, float, bool]:
        """Ensure sampled weighted contribution of control_cost is non-positive."""

        expr = str(formula).strip()
        if not expr:
            return formula, weight, False

        allowed_vars = set(primitive_interface.get("allowed_formula_variables", [])) or cls.DEFAULT_ALLOWED_VARS
        samples = cls._sample_variables(allowed_vars)

        values: List[float] = []
        for variables in samples:
            try:
                values.append(float(weight) * float(safe_eval_formula(expr, variables=variables)))
            except Exception:
                continue

        if not values:
            return formula, weight, False

        # Already non-positive under samples.
        if max(values) <= 1e-9 and float(weight) >= 0:
            return formula, float(weight), False

        # Canonical convention: keep weight non-negative, put penalty sign in formula.
        new_weight = abs(float(weight)) if abs(float(weight)) > 0 else 1.0
        new_formula = f"-abs(({expr}))"
        return new_formula, new_weight, True

    @staticmethod
    def _sample_variables(allowed_vars: set[str]) -> List[Dict[str, Any]]:
        def base() -> Dict[str, Any]:
            d: Dict[str, Any] = {name: 0.0 for name in allowed_vars}
            for k in ["left_contact", "right_contact", "contact", "both_contact"]:
                if k in d:
                    d[k] = False
            return d

        samples = []
        for main_engine, side_engine in [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (1.0, -1.0),
        ]:
            x = base()
            if "main_engine" in x:
                x["main_engine"] = main_engine
            if "side_engine" in x:
                x["side_engine"] = side_engine
            samples.append(x)
        return samples

    @staticmethod
    def _build_blueprint_role_hints(blueprint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        hints: Dict[str, Dict[str, Any]] = {}
        for item in blueprint.get("component_blueprint", []) or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            hints[name] = dict(item)
            if name.startswith("r_"):
                hints[name[2:]] = dict(item)
            else:
                hints[f"r_{name}"] = dict(item)
        return hints

    @staticmethod
    def _match_hint(name: str, hints: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
        if not name:
            return None
        if name in hints:
            return hints[name]
        if name.startswith("r_") and name[2:] in hints:
            return hints[name[2:]]
        alt = f"r_{name}"
        if alt in hints:
            return hints[alt]
        return None

    @staticmethod
    def _timing_from_role_phase(role: str | None, phase: str | None) -> str | None:
        if role == "terminal_success":
            return "sparse_event"
        if phase == "completion":
            return "sparse_event"
        return "dense"

    @staticmethod
    def _channel_from_role_phase(role: str | None, phase: str | None) -> str | None:
        if role == "dense_guidance":
            return "progress"
        if role == "stability_quality":
            return "stability"
        if role == "control_cost":
            return "control"
        if role == "terminal_success":
            return "completion"
        if role == "safety_constraint":
            return "safety"
        if phase:
            return str(phase)
        return None
