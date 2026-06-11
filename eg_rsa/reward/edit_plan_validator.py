from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eg_rsa.reward.formula_validator import FormulaValidator
from eg_rsa.reward.operators import RewardEditOperatorApplier
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.reward.safe_formula_eval import safe_eval_formula


@dataclass
class EditPlanValidationResult:
    valid_edits: List[Dict[str, Any]] = field(default_factory=list)
    rejected_edits: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.valid_edits) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "valid_edits": self.valid_edits,
            "rejected_edits": self.rejected_edits,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class EditPlanValidator:
    """Validate and normalize edit plans before schema mutation."""

    DEFAULT_ALLOWED_VARS = {
        "x", "y", "vx", "vy", "angle", "angular_velocity",
        "left_contact", "right_contact", "main_engine", "side_engine",
    }
    DEFAULT_ALLOWED_FUNCS = {"abs", "min", "max", "clip", "sqrt", "exp", "tanh"}

    @classmethod
    def validate(
        cls,
        schema: RewardSchema,
        edit_plan: List[Dict[str, Any]],
        structural_context: Optional[Dict[str, Any]] = None,
    ) -> EditPlanValidationResult:
        structural_context = structural_context or {}
        result = EditPlanValidationResult()

        if not isinstance(edit_plan, list):
            result.errors.append("edit_plan must be a list")
            return result

        for idx, edit in enumerate(edit_plan):
            normalized, notes = cls._normalize_edit(edit, structural_context)
            result.warnings.extend(notes)
            ok, error = cls._validate_one(schema, normalized, structural_context)
            if ok:
                result.valid_edits.append(normalized)
            else:
                rejected = dict(normalized) if isinstance(normalized, dict) else {"raw_edit": normalized}
                rejected["index"] = idx
                result.rejected_edits.append(rejected)
                result.errors.append(error)

        return result

    @classmethod
    def safe_fallback(cls, schema: RewardSchema, diagnostics: Dict[str, Any]) -> List[Dict[str, Any]]:
        target = diagnostics.get("dominant_component")
        modes = set(diagnostics.get("failure_modes", []))

        if target and schema.get_component(target):
            if "single_component_dominance" in modes:
                return [{"operator": "decrease_weight", "target": target, "factor": 0.5}]
            return [{"operator": "clip_component", "target": target, "clip": [-1.0, 1.0]}]

        for component in schema.components:
            if component.enabled:
                return [{"operator": "clip_component", "target": component.name, "clip": [-1.0, 1.0]}]

        return []

    @classmethod
    def _normalize_edit(cls, edit: Any, structural_context: Dict[str, Any]) -> Tuple[Any, List[str]]:
        if not isinstance(edit, dict):
            return edit, []

        op = edit.get("operator") or edit.get("op")

        if op in {"add_formula_component", "add_conditional_formula_component", "add_action_penalty"}:
            return cls._normalize_add_formula_alias(edit, op)

        if op == "add_event_predicate":
            return cls._normalize_add_event_predicate(edit)

        if op == "add_component":
            return cls._normalize_add_component(edit)

        if op == "add_event_rule":
            return cls._normalize_add_event_rule(edit, structural_context)

        if op in {"replace_formula", "replace_condition"}:
            normalized = dict(edit)
            normalized["operator"] = op
            return normalized, []

        return edit, []

    @staticmethod
    def _normalize_add_formula_alias(edit: Dict[str, Any], op: str) -> Tuple[Dict[str, Any], List[str]]:
        notes: List[str] = []
        normalized = dict(edit)
        component = dict(normalized.get("component", {}) or {})

        if op == "add_formula_component":
            component["type"] = "formula_component"
        elif op == "add_conditional_formula_component":
            component["type"] = "conditional_formula_component"
        elif op == "add_action_penalty":
            component["type"] = "action_penalty"

        params = dict(component.get("params", {}) or {})

        if "formula" in component and "formula" not in params:
            params["formula"] = component["formula"]
            notes.append("Moved component.formula into component.params.formula.")

        if "condition" in component and "condition" not in params:
            params["condition"] = component["condition"]
            notes.append("Moved component.condition into component.params.condition.")

        component["params"] = params
        component.setdefault("inputs", [])
        component.setdefault("enabled", True)

        normalized["component"] = component
        normalized["operator"] = "add_component"
        return normalized, notes

    @staticmethod
    def _normalize_add_event_predicate(edit: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        normalized = dict(edit)
        rule = dict(normalized.get("event_rule", {}) or {})
        rule["type"] = "event_predicate"
        rule.setdefault("enabled", True)
        rule.setdefault("one_time", True)
        condition = rule.get("condition", {})
        if isinstance(condition, str):
            rule["condition"] = {"expression": condition, "duration_steps": int(rule.get("duration_steps", 1) or 1)}
        elif isinstance(condition, dict):
            condition = dict(condition)
            condition.setdefault("duration_steps", int(rule.get("duration_steps", condition.get("duration_steps", 1)) or 1))
            rule["condition"] = condition
        normalized["event_rule"] = rule
        normalized["operator"] = "add_event_rule"
        return normalized, []

    @classmethod
    def _normalize_add_event_rule(cls, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        notes: List[str] = []
        normalized = dict(edit)
        rule = normalized.get("event_rule")

        if not isinstance(rule, dict):
            return normalized, notes

        rule = dict(rule)
        available_events = set(structural_context.get("available_events", []))
        event_name = rule.get("event") or rule.get("event_name") or rule.get("metric")

        if ("condition" not in rule or not isinstance(rule.get("condition"), dict)) and event_name:
            if not available_events or event_name in available_events:
                rule["condition"] = {event_name: True}
                rule.setdefault("name", cls._safe_rule_name(event_name, bool(rule.get("one_time", True))))
                rule.setdefault("type", "event_bonus")
                rule.setdefault("enabled", True)
                rule.setdefault("one_time", True)
                notes.append(f"Normalized compact add_event_rule for event {event_name!r}.")

        if "duration_steps" in rule:
            condition = dict(rule.get("condition", {}))
            try:
                duration = int(rule.get("duration_steps"))
                condition["duration_steps"] = max(1, duration)
                notes.append("Moved event_rule.duration_steps into event_rule.condition.duration_steps.")
            except (TypeError, ValueError):
                notes.append("Ignored invalid event_rule.duration_steps during normalization.")
            rule.pop("duration_steps", None)
            rule["condition"] = condition

        if isinstance(rule.get("condition"), dict) and "duration_steps" in rule["condition"]:
            condition = dict(rule["condition"])
            try:
                condition["duration_steps"] = max(1, int(condition["duration_steps"]))
            except (TypeError, ValueError):
                condition.pop("duration_steps", None)
                notes.append("Removed invalid condition.duration_steps during normalization.")
            rule["condition"] = condition

        rule.setdefault("type", "event_bonus")
        rule.setdefault("enabled", True)
        rule.setdefault("one_time", False)

        normalized["event_rule"] = rule
        normalized["operator"] = "add_event_rule"
        return normalized, notes

    @staticmethod
    def _normalize_add_component(edit: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        notes: List[str] = []
        normalized = dict(edit)
        component = normalized.get("component")

        if not isinstance(component, dict):
            return normalized, notes

        component = dict(component)
        component_type = component.get("type")
        params = dict(component.get("params", {}) or {})

        if component_type in RewardEditOperatorApplier.FORMULA_COMPONENT_TYPES:
            if "formula" in component and "formula" not in params:
                params["formula"] = component["formula"]
                notes.append("Moved component.formula into component.params.formula.")
            if "condition" in component and "condition" not in params:
                params["condition"] = component["condition"]
                notes.append("Moved component.condition into component.params.condition.")
            component["params"] = params
            component.setdefault("inputs", [])
            component.setdefault("enabled", True)

        elif component_type in RewardEditOperatorApplier.METRIC_COMPONENT_TYPES:
            if "metric" not in params and component.get("metric"):
                params["metric"] = component.get("metric")
                notes.append("Moved component.metric into component.params.metric.")
            if component_type == "metric_stagnation_penalty":
                if int(params.get("window", 0) or 0) <= 0:
                    params["window"] = 50
                    notes.append("Filled metric_stagnation_penalty params.window with default 50.")
                try:
                    threshold = float(params.get("threshold", 1e-3))
                except (TypeError, ValueError):
                    threshold = 1e-3
                params["threshold"] = max(0.0, threshold)
            component["params"] = params
            component.setdefault("inputs", [])
            component.setdefault("enabled", True)
            if component.get("clip") is None and component_type in {"metric_value", "metric_delta", "metric_threshold_bonus"}:
                component["clip"] = [0.0, 1.0]
            if component.get("clip") is None and component_type == "metric_stagnation_penalty":
                component["clip"] = [-1.0, 0.0]

        normalized["component"] = component
        normalized["operator"] = "add_component"
        return normalized, notes

    @classmethod
    def _validate_one(cls, schema: RewardSchema, edit: Any, structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(edit, dict):
            return False, "edit must be a dict"

        op = edit.get("operator") or edit.get("op")
        if op not in RewardEditOperatorApplier.ALLOWED_OPERATORS:
            return False, f"unsupported operator: {op}"

        if op in {"increase_weight", "decrease_weight", "clip_component", "disable_component", "convert_to_one_time_event", "add_duration_condition", "reshape_sparse_to_dense", "replace_formula", "replace_condition"}:
            target = edit.get("target")
            if not target:
                return False, f"{op} requires target"
            if schema.get_component(target) is None and schema.get_event_rule(target) is None:
                return False, f"target not found: {target}"

        if op == "increase_weight" and float(edit.get("factor", 0.0)) <= 1.0:
            return False, "increase_weight requires factor > 1"

        if op == "decrease_weight":
            factor = float(edit.get("factor", 0.0))
            if not 0.0 < factor < 1.0:
                return False, "decrease_weight requires 0 < factor < 1"

        if op == "clip_component":
            clip = edit.get("clip")
            if not isinstance(clip, list) or len(clip) != 2 or float(clip[0]) > float(clip[1]):
                return False, "clip_component requires clip=[min,max]"
            if schema.get_component(edit.get("target")) is None:
                return False, "clip_component target must be a component"

        if op == "replace_formula":
            return cls._validate_replace_formula(schema, edit, structural_context)

        if op == "replace_condition":
            return cls._validate_replace_condition(schema, edit, structural_context)

        if op == "convert_to_one_time_event":
            target = edit.get("target")
            component = schema.get_component(target)
            rule = schema.get_event_rule(target)
            if component is None and rule is None:
                return False, f"target not found: {target}"
            if component is not None and component.type != "event_bonus":
                return False, "convert_to_one_time_event requires an event_bonus component or event rule"

        if op == "add_component":
            return cls._validate_add_component(schema, edit, structural_context)

        if op == "add_event_rule":
            return cls._validate_add_event_rule(schema, edit, structural_context)

        if op == "add_duration_condition":
            if schema.get_event_rule(edit.get("target")) is None:
                return False, "add_duration_condition target must be an event rule"
            if int(edit.get("duration_steps", 0)) <= 0:
                return False, "duration_steps must be positive"

        if op == "reshape_sparse_to_dense":
            new_type = edit.get("new_type")
            if new_type not in RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES:
                return False, f"reshape_sparse_to_dense new_type must be one of {sorted(RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES)}"

        return True, ""

    @classmethod
    def _validate_replace_formula(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        target = edit.get("target")
        component = schema.get_component(target)
        if component is None:
            return False, "replace_formula target must be a component"
        if component.type not in RewardEditOperatorApplier.FORMULA_COMPONENT_TYPES:
            return False, f"replace_formula target must be formula-native component, got {component.type}"
        formula = edit.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            return False, "replace_formula requires non-empty formula"
        err = cls._validate_formula_text(formula, structural_context)
        if err:
            return False, err

        contract_error = cls._validate_replace_formula_contract(schema, component, formula)
        if contract_error:
            return False, contract_error

        return True, ""

    @classmethod
    def _validate_replace_formula_contract(cls, schema: RewardSchema, component: Any, formula: str) -> str:
        """Minimal schema-language contract for formula-native replacement.

        This is intentionally not task-specific. It only prevents an edit from
        silently deleting the sole process signal by replacing it with a constant.
        More complex quality judgments should be measured by the next rollout
        and OutcomeAcceptor/Memory, not by hand-coded environment rules.
        """

        expr = str(formula or "").strip()
        if not expr:
            return "replace_formula contract violation: empty formula"

        role = (
            getattr(component, "semantic_role", None)
            or getattr(component, "role", None)
            or getattr(component, "metadata", {}).get("semantic_role", None)
            if hasattr(component, "metadata")
            else None
        )

        # RewardComponent may store arbitrary fields only through params in some
        # versions, so also inspect params.
        params = getattr(component, "params", {}) or {}
        if role is None:
            role = params.get("semantic_role") or params.get("role")

        is_constant = cls._is_constant_formula(expr)

        if str(role) == "dense_guidance":
            dense_guidance = []
            for c in schema.components:
                c_role = (
                    getattr(c, "semantic_role", None)
                    or getattr(c, "role", None)
                    or (getattr(c, "params", {}) or {}).get("semantic_role")
                    or (getattr(c, "params", {}) or {}).get("role")
                )
                if str(c_role) == "dense_guidance" and getattr(c, "enabled", True):
                    dense_guidance.append(c.name)

            if len(dense_guidance) <= 1 and is_constant:
                return (
                    "replace_formula contract violation: cannot replace the only "
                    "enabled dense_guidance component with a constant formula. "
                    "Use reshape formula or explicit disable_component instead."
                )

        return ""

    @staticmethod
    def _is_constant_formula(expr: str) -> bool:
        # Deliberately simple and environment-agnostic: if the expression contains
        # no alphabetic identifier except allowed function names and numeric
        # syntax, treat it as constant.
        lowered = expr.strip().lower()
        if lowered in {"0", "0.0", "1", "1.0", "-1", "-1.0"}:
            return True

        identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
        function_names = {"abs", "min", "max", "clip", "sqrt", "exp", "tanh", "and", "or", "not", "true", "false"}
        identifiers = {x for x in identifiers if x.lower() not in function_names}
        return len(identifiers) == 0

    @classmethod
    def _validate_replace_condition(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        target = edit.get("target")
        component = schema.get_component(target)
        rule = schema.get_event_rule(target)
        condition = edit.get("condition")

        if component is not None:
            if component.type != "conditional_formula_component":
                return False, "replace_condition component target must be conditional_formula_component"
            if not isinstance(condition, str) or not condition.strip():
                return False, "replace_condition for component requires string condition"
            err = cls._validate_formula_text(condition, structural_context)
            if err:
                return False, err
            return True, ""

        if rule is not None:
            if rule.type != "event_predicate":
                return False, "replace_condition event target must be event_predicate"
            expr = condition.get("expression") if isinstance(condition, dict) else condition
            if not isinstance(expr, str) or not expr.strip():
                return False, "replace_condition for event_predicate requires expression"
            err = cls._validate_formula_text(expr, structural_context)
            if err:
                return False, err
            return True, ""

        return False, "replace_condition target not found"

    @classmethod
    def _validate_add_component(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        component = edit.get("component")
        if not isinstance(component, dict):
            return False, "add_component requires component dict"

        name = component.get("name")
        if not name:
            return False, "new component requires name"

        if schema.get_component(name) or schema.get_event_rule(name):
            return False, f"reward item already exists: {name}"

        component_type = component.get("type")
        if component_type not in RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES:
            return False, f"add_component type {component_type!r} is not allowed; use one of {sorted(RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES)}"

        if component_type in RewardEditOperatorApplier.FORMULA_COMPONENT_TYPES:
            error = cls._validate_formula_component(component, structural_context)
        else:
            error = cls._validate_metric_component(component, structural_context)

        if error:
            return False, error

        return True, ""

    @classmethod
    def _validate_add_event_rule(cls, schema: RewardSchema, edit: Dict[str, Any], structural_context: Dict[str, Any]) -> Tuple[bool, str]:
        rule = edit.get("event_rule")
        if not isinstance(rule, dict):
            return False, "add_event_rule requires event_rule dict"

        name = rule.get("name")
        if not name:
            return False, "event_rule requires name"

        if schema.get_component(name) or schema.get_event_rule(name):
            return False, f"reward item already exists: {name}"

        rule_type = rule.get("type")
        condition = rule.get("condition")

        if rule_type not in {"event_bonus", "event_predicate"}:
            return False, "event_rule type must be event_bonus or event_predicate"

        if not isinstance(condition, dict) or not condition:
            return False, "event_rule condition must be non-empty dict"

        if rule_type == "event_predicate":
            expr = condition.get("expression") or condition.get("formula")
            if not isinstance(expr, str) or not expr.strip():
                return False, "event_predicate requires condition.expression"
            err = cls._validate_formula_text(expr, structural_context)
            if err:
                return False, f"event_predicate condition is unsafe: {err}"
            return True, ""

        unknown_events = cls._unknown_condition_events(condition, structural_context)
        if unknown_events:
            return False, f"event_rule condition references unknown events: {unknown_events}"

        return True, ""

    @classmethod
    def _validate_formula_component(cls, component: Dict[str, Any], structural_context: Dict[str, Any]) -> str:
        component_type = component.get("type")
        params = component.get("params", {}) or {}

        if component_type == "action_penalty" and not params.get("formula"):
            return ""

        formula = params.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            return f"{component_type} requires params.formula"

        err = cls._validate_formula_text(formula, structural_context)
        if err:
            return f"{component_type} formula is unsafe: {err}"

        if component_type == "conditional_formula_component":
            condition = params.get("condition")
            if not isinstance(condition, str) or not condition.strip():
                return "conditional_formula_component requires params.condition"
            err = cls._validate_formula_text(condition, structural_context)
            if err:
                return f"conditional_formula_component condition is unsafe: {err}"

        if component_type == "action_penalty":
            sign_error = cls._action_penalty_sign_error(component, structural_context)
            if sign_error:
                return sign_error

        return ""

    @classmethod
    def _validate_formula_text(cls, expr: str, structural_context: Dict[str, Any]) -> str:
        allowed_vars = set(structural_context.get("allowed_formula_variables", [])) or cls.DEFAULT_ALLOWED_VARS
        allowed_funcs = set(structural_context.get("allowed_formula_functions", [])) or cls.DEFAULT_ALLOWED_FUNCS
        validation = FormulaValidator.validate_expression(expr, allowed_vars, allowed_funcs)
        if not validation.ok:
            return str(validation.errors)
        return ""

    @classmethod
    def _action_penalty_sign_error(cls, component: Dict[str, Any], structural_context: Dict[str, Any]) -> str:
        params = component.get("params", {}) or {}
        formula = params.get("formula")
        if not formula:
            return ""
        try:
            weight = float(component.get("weight", 1.0))
        except Exception:
            return ""

        allowed_vars = set(structural_context.get("allowed_formula_variables", [])) or cls.DEFAULT_ALLOWED_VARS
        base = {var: 0.0 for var in allowed_vars}
        samples = [
            {"main_engine": 0.0, "side_engine": 0.0},
            {"main_engine": 1.0, "side_engine": 0.0},
            {"main_engine": 0.0, "side_engine": 1.0},
            {"main_engine": 0.0, "side_engine": -1.0},
            {"main_engine": 1.0, "side_engine": 1.0},
            {"main_engine": 1.0, "side_engine": -1.0},
        ]

        for sample in samples:
            variables = dict(base)
            variables.update(sample)
            try:
                raw = float(safe_eval_formula(str(formula), variables=variables))
            except Exception:
                continue
            value = weight * raw
            if value > 1e-9:
                return (
                    f"action_penalty {component.get('name')} can produce positive reward "
                    f"under sampled action {sample}: {value:.6g}"
                )

        return ""

    @staticmethod
    def _validate_metric_component(component: Dict[str, Any], structural_context: Dict[str, Any]) -> str:
        component_type = component.get("type")
        if component_type not in RewardEditOperatorApplier.ADDABLE_COMPONENT_TYPES:
            return f"unsupported metric component type: {component_type}"

        params = component.get("params", {})
        metric = params.get("metric")

        if not metric:
            return f"{component_type} requires params.metric"

        available_metrics = set(structural_context.get("available_task_metrics", []))
        if available_metrics and metric not in available_metrics:
            return f"{component_type} references unknown task metric: {metric}"

        if component_type == "metric_threshold_bonus":
            if "threshold" not in params:
                return "metric_threshold_bonus requires params.threshold"
            if params.get("direction", "ge") not in {"ge", "le"}:
                return "metric_threshold_bonus params.direction must be ge or le"

        if component_type == "metric_stagnation_penalty":
            if int(params.get("window", 0)) <= 0:
                return "metric_stagnation_penalty requires positive params.window"
            if float(params.get("threshold", -1.0)) < 0:
                return "metric_stagnation_penalty requires non-negative params.threshold"

        return ""

    @staticmethod
    def _unknown_condition_events(condition: Dict[str, Any], structural_context: Dict[str, Any]) -> List[str]:
        available_events = set(structural_context.get("available_events", []))
        if not available_events:
            return []
        return [key for key in condition.keys() if key != "duration_steps" and key not in available_events]

    @staticmethod
    def _safe_rule_name(event_name: str, one_time: bool) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(event_name)).strip("_")
        suffix = "once" if one_time else "event"
        return f"r_{safe}_{suffix}"
