from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from eg_rsa.reward.formula_ast import FormulaAST


class SchemaCanonicalizer:
    """Canonicalize reward schema into AST-first IR.

    This is intentionally strict:
      - It preserves formula_ast / condition_ast.
      - It mirrors top-level AST fields into params.
      - It does NOT try to parse or repair LLM formula strings.
    """

    @classmethod
    def canonicalize_bootstrap_result(
        cls,
        bootstrap_result: Dict[str, Any],
        primitive_interface: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        result = copy.deepcopy(bootstrap_result or {})
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
        primitive_interface: Dict[str, Any] = None,
        reward_blueprint: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        primitive_interface = primitive_interface or {}
        reward_blueprint = reward_blueprint or {}

        data = copy.deepcopy(schema or {})
        data.setdefault("version", 2)
        data.setdefault("metadata", {})
        data.setdefault("components", [])
        data.setdefault("event_rules", [])

        if primitive_interface.get("action_mapping"):
            data["metadata"]["action_mapping"] = primitive_interface.get("action_mapping")
        if primitive_interface.get("action_variables"):
            data["metadata"]["action_variables"] = primitive_interface.get("action_variables")
        if primitive_interface.get("allowed_formula_variables"):
            data["metadata"]["allowed_formula_variables"] = primitive_interface.get("allowed_formula_variables")
        if primitive_interface.get("allowed_formula_functions"):
            data["metadata"]["allowed_formula_functions"] = primitive_interface.get("allowed_formula_functions")
        data["metadata"]["formula_ir"] = "ast"

        notes: List[str] = []
        errors: List[str] = []

        role_hints = cls._build_blueprint_role_hints(reward_blueprint)

        canonical_events: List[Dict[str, Any]] = []
        for raw_rule in data.get("event_rules", []) or []:
            if not isinstance(raw_rule, dict):
                errors.append("Non-dict event_rule dropped during canonicalization")
                continue
            canonical_events.append(cls._canonicalize_event_rule(raw_rule, role_hints, notes))

        canonical_components: List[Dict[str, Any]] = []
        for raw_component in data.get("components", []) or []:
            if not isinstance(raw_component, dict):
                errors.append("Non-dict component dropped during canonicalization")
                continue
            canonical_components.append(cls._canonicalize_component(raw_component, role_hints, notes))

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
            "formula_ir": "ast",
        }
        return data, report

    @classmethod
    def _canonicalize_component(
        cls,
        component: Dict[str, Any],
        role_hints: Dict[str, Dict[str, Any]],
        notes: List[str],
    ) -> Dict[str, Any]:
        comp = copy.deepcopy(component)
        name = str(comp.get("name", ""))

        comp.setdefault("enabled", True)
        comp.setdefault("weight", 1.0)
        comp.setdefault("params", {})
        comp.setdefault("inputs", [])

        params = dict(comp.get("params", {}) or {})

        if not comp.get("semantic_role") and comp.get("role"):
            comp["semantic_role"] = comp.get("role")
            notes.append(f"Component {name}: copied role -> semantic_role.")

        hint = cls._match_hint(name, role_hints)
        if hint and not comp.get("semantic_role") and hint.get("role"):
            comp["semantic_role"] = hint["role"]
            notes.append(f"Component {name}: filled semantic_role from blueprint.")
        if hint and not comp.get("reward_timing"):
            comp["reward_timing"] = cls._timing_from_role_phase(hint.get("role"), hint.get("phase"))
        if hint and not comp.get("behavior_channel"):
            comp["behavior_channel"] = cls._channel_from_role_phase(hint.get("role"), hint.get("phase"))

        ctype = comp.get("type")

        if ctype == "action_penalty" and (comp.get("formula_ast") or params.get("formula_ast")):
            comp["type"] = "formula_component"
            comp.setdefault("semantic_role", "control_cost")
            comp.setdefault("reward_timing", "dense")
            comp.setdefault("behavior_channel", "control")
            ctype = "formula_component"
            notes.append(f"Component {name}: converted AST action_penalty to formula_component.")

        if "formula_ast" in comp and "formula_ast" not in params:
            params["formula_ast"] = FormulaAST.normalize(comp["formula_ast"])
        elif "formula_ast" in params:
            params["formula_ast"] = FormulaAST.normalize(params["formula_ast"])
            comp["formula_ast"] = params["formula_ast"]

        if "condition_ast" in comp and "condition_ast" not in params:
            params["condition_ast"] = FormulaAST.normalize(comp["condition_ast"])
        elif "condition_ast" in params:
            params["condition_ast"] = FormulaAST.normalize(params["condition_ast"])
            comp["condition_ast"] = params["condition_ast"]

        if ctype in {"formula_component", "conditional_formula_component"} and "formula_ast" in params:
            comp["formula_ast"] = params["formula_ast"]
        if ctype == "conditional_formula_component" and "condition_ast" in params:
            comp["condition_ast"] = params["condition_ast"]

        comp["params"] = params
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

        if "reward" in out:
            try:
                out["weight"] = float(out.get("reward"))
                notes.append(f"Event rule {name}: moved reward into weight.")
            except Exception:
                notes.append(f"Event rule {name}: ignored non-numeric reward field.")
            out.pop("reward", None)
        else:
            out.setdefault("weight", 1.0)

        if not out.get("semantic_role") and out.get("role"):
            out["semantic_role"] = out.get("role")
            notes.append(f"Event rule {name}: copied role -> semantic_role.")

        hint = cls._match_hint(name, role_hints)
        if hint and not out.get("semantic_role") and hint.get("role"):
            out["semantic_role"] = hint["role"]
            notes.append(f"Event rule {name}: filled semantic_role from blueprint.")

        if not out.get("semantic_role") and bool(out.get("one_time", False)):
            try:
                w = float(out.get("weight", 1.0) or 1.0)
                out["semantic_role"] = "terminal_success" if w >= 0.0 else "safety_constraint"
                notes.append(f"Event rule {name}: inferred semantic_role from one_time sign.")
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
        condition = dict(condition or {}) if isinstance(condition, dict) else {}

        ast_node = (
            out.get("condition_ast")
            or out.get("expr_ast")
            or condition.get("expr_ast")
            or condition.get("condition_ast")
        )
        if ast_node is not None:
            condition["expr_ast"] = FormulaAST.normalize(ast_node)

        condition.setdefault("duration_steps", int(out.get("duration_steps", condition.get("duration_steps", 1)) or 1))
        out["condition"] = condition

        out.pop("condition_ast", None)
        out.pop("expr_ast", None)
        out.pop("expression", None)
        out.pop("formula", None)
        out.pop("duration_steps", None)
        return out

    @staticmethod
    def _build_blueprint_role_hints(blueprint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        hints: Dict[str, Dict[str, Any]] = {}
        for item in blueprint.get("component_blueprint", []) or []:
            if isinstance(item, dict) and item.get("name"):
                hints[str(item["name"])] = dict(item)
        return hints

    @staticmethod
    def _match_hint(name: str, hints: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if name in hints:
            return hints[name]
        lname = str(name).lower()
        for key, value in hints.items():
            if str(key).lower() == lname:
                return value
        return {}

    @staticmethod
    def _timing_from_role_phase(role: str, phase: str) -> str:
        if role == "terminal_success":
            return "sparse_event"
        return "dense"

    @staticmethod
    def _channel_from_role_phase(role: str, phase: str) -> str:
        if role == "terminal_success":
            return "completion"
        if role == "control_cost":
            return "control"
        if role == "stability_quality":
            return "stability"
        if role == "safety_constraint":
            return "safety"
        return "progress"
