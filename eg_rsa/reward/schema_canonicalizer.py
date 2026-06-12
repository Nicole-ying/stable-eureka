from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from eg_rsa.reward.formula_ast import FormulaAST


class SchemaCanonicalizer:
    """Canonicalize reward schema into AST-first IR.

    This is intentionally strict about executable formulas:
      - It preserves formula_ast / condition_ast.
      - It mirrors top-level AST fields into params.
      - It does NOT try to parse or repair LLM formula strings.

    It is conservative about non-semantic formatting defects:
      - Empty or malformed clip ranges are normalized to role-based defaults when
        doing so is unambiguous, because SafeRewardCompiler requires clip to be
        None or a valid [low, high] pair.
      - Missing event-rule names are filled with stable role/index-based names
        when the rule has an executable predicate.
      - Non-executable event rules are dropped. Event rules are optional in
        source-aware bootstrap; a malformed sparse event should not block a valid
        dense reward schema from entering training. Missing formulas in dense
        components remain validation errors.
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
        for event_index, raw_rule in enumerate(data.get("event_rules", []) or []):
            if not isinstance(raw_rule, dict):
                notes.append("Non-dict event_rule dropped during canonicalization.")
                continue
            canonical_rule = cls._canonicalize_event_rule(raw_rule, role_hints, notes, event_index)
            if canonical_rule is not None:
                canonical_events.append(canonical_rule)

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

        cls._canonicalize_clip(comp, notes)

        comp["params"] = params
        return comp

    @classmethod
    def _canonicalize_event_rule(
        cls,
        rule: Dict[str, Any],
        role_hints: Dict[str, Dict[str, Any]],
        notes: List[str],
        event_index: int = 0,
    ) -> Optional[Dict[str, Any]]:
        out = copy.deepcopy(rule)
        name = str(out.get("name", "") or "")

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

        condition = out.get("condition", {})
        condition = dict(condition or {}) if isinstance(condition, dict) else {}

        ast_node = (
            out.get("condition_ast")
            or out.get("expr_ast")
            or condition.get("expr_ast")
            or condition.get("condition_ast")
        )
        if ast_node is None:
            drop_name = name or str(out.get("semantic_role") or f"event_{event_index}")
            notes.append(
                f"Event rule {drop_name}: dropped because it has no executable condition.expr_ast. "
                "Source-aware bootstrap treats event_rules as optional."
            )
            return None
        condition["expr_ast"] = FormulaAST.normalize(ast_node)

        if not out.get("name"):
            role_for_name = str(out.get("semantic_role") or "event").lower()
            out["name"] = cls._default_event_name(role_for_name, event_index)
            name = str(out["name"])
            notes.append(f"Event rule missing name: filled name -> {name}.")

        out.setdefault("reward_timing", "sparse_event")
        if not out.get("behavior_channel"):
            role = out.get("semantic_role")
            if role == "terminal_success":
                out["behavior_channel"] = "completion"
            elif role == "safety_constraint":
                out["behavior_channel"] = "safety"

        condition.setdefault("duration_steps", int(out.get("duration_steps", condition.get("duration_steps", 1)) or 1))
        out["condition"] = condition

        out.pop("condition_ast", None)
        out.pop("expr_ast", None)
        out.pop("expression", None)
        out.pop("formula", None)
        out.pop("duration_steps", None)
        return out

    @staticmethod
    def _default_event_name(role: str, event_index: int) -> str:
        safe_role = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(role or "event").lower()).strip("_")
        if not safe_role:
            safe_role = "event"
        return f"r_{safe_role}_event_{int(event_index):02d}"

    @classmethod
    def _canonicalize_clip(cls, comp: Dict[str, Any], notes: List[str]) -> None:
        """Normalize LLM clip field into compiler-compatible form.

        SafeRewardCompiler accepts clip=None or a two-value numeric range. LLMs
        sometimes emit clip: [] when unsure. Empty clip does not carry reward
        semantics, so we repair it using the component semantic role.
        """
        if "clip" not in comp:
            return

        name = str(comp.get("name", ""))
        raw_clip = comp.get("clip")
        normalized = cls._parse_clip(raw_clip)
        if normalized is not None:
            if normalized != raw_clip:
                notes.append(f"Component {name}: normalized clip {raw_clip!r} -> {normalized!r}.")
            comp["clip"] = normalized
            return

        default_clip = cls._default_clip_for_role(comp.get("semantic_role"), comp.get("name"))
        if default_clip is not None:
            comp["clip"] = default_clip
            notes.append(
                f"Component {name}: replaced invalid clip {raw_clip!r} with role-based default {default_clip!r}."
            )
        else:
            comp.pop("clip", None)
            notes.append(f"Component {name}: removed invalid clip {raw_clip!r}; compiler will treat it as unclipped.")

    @staticmethod
    def _parse_clip(raw_clip: Any) -> Optional[List[float]]:
        if not isinstance(raw_clip, (list, tuple)) or len(raw_clip) != 2:
            return None
        try:
            low = float(raw_clip[0])
            high = float(raw_clip[1])
        except Exception:
            return None
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            return None
        return [low, high]

    @staticmethod
    def _default_clip_for_role(role: Any, name: Any = "") -> Optional[List[float]]:
        role = str(role or "").lower()
        lname = str(name or "").lower()
        if role in {"control_cost", "safety_constraint"}:
            return [-1.0, 0.0]
        if role in {"dense_guidance", "stability_quality", "terminal_success"}:
            return [0.0, 1.0]
        if any(token in lname for token in ["cost", "penalty", "unsafe", "crash", "fuel"]):
            return [-1.0, 0.0]
        if any(token in lname for token in ["safe", "stability", "stable", "progress", "guidance"]):
            return [0.0, 1.0]
        return None

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
