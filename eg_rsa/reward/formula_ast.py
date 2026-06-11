from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set


@dataclass
class FormulaASTValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class FormulaASTError(ValueError):
    pass


class FormulaAST:
    """Strongly typed reward expression IR.

    Accepted leaf nodes:
      {"var": "x"}
      {"const": 1.0}
      {"bool": true}

    Accepted operator nodes:
      {"op": "add", "args": [...]}
      {"op": "sub", "left": ..., "right": ...}
      {"op": "mul", "args": [...]}
      {"op": "div", "left": ..., "right": ...}
      {"op": "neg", "arg": ...}
      {"op": "abs", "arg": ...}
      {"op": "min", "args": [...]}
      {"op": "max", "args": [...]}
      {"op": "clip", "args": [value, low, high]}
      {"op": "and", "args": [...]}
      {"op": "or", "args": [...]}
      {"op": "not", "arg": ...}
      {"op": "lt|le|gt|ge|eq|ne", "left": ..., "right": ...}

    This module intentionally does not parse LLM-written formula strings.
    """

    NUMERIC_NARY = {"add", "mul", "min", "max"}
    NUMERIC_BINARY = {"sub", "div", "pow"}
    NUMERIC_UNARY = {"neg", "abs", "sqrt", "exp", "tanh"}
    BOOL_NARY = {"and", "or"}
    BOOL_UNARY = {"not"}
    COMPARE = {"lt", "le", "gt", "ge", "eq", "ne"}
    SPECIAL = {"clip"}

    OP_ALIASES = {
        "+": "add",
        "sum": "add",
        "-": "sub",
        "*": "mul",
        "product": "mul",
        "/": "div",
        "**": "pow",
        "^": "pow",
        "negative": "neg",
        "AND": "and",
        "OR": "or",
        "NOT": "not",
        "all": "and",
        "any": "or",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
        "==": "eq",
        "!=": "ne",
    }

    @classmethod
    def normalize(cls, node: Any) -> Any:
        if isinstance(node, (int, float, bool)):
            return {"bool": bool(node)} if isinstance(node, bool) else {"const": float(node)}

        if not isinstance(node, dict):
            return node

        out = dict(node)

        if "variable" in out and "var" not in out:
            out["var"] = out.pop("variable")
        if "name" in out and "var" not in out and "op" not in out:
            out["var"] = out.pop("name")
        if "value" in out and "const" not in out and "bool" not in out and "op" not in out:
            value = out.pop("value")
            if isinstance(value, bool):
                out["bool"] = value
            else:
                out["const"] = value

        if "op" in out:
            op = str(out.get("op"))
            op = cls.OP_ALIASES.get(op, op).lower()
            out["op"] = op

        for key in ("arg", "left", "right"):
            if key in out:
                out[key] = cls.normalize(out[key])

        if "args" in out and isinstance(out["args"], list):
            out["args"] = [cls.normalize(x) for x in out["args"]]

        return out

    @classmethod
    def validate(
        cls,
        node: Any,
        allowed_variables: Iterable[str],
    ) -> FormulaASTValidationResult:
        allowed = set(allowed_variables or [])
        errors: List[str] = []
        cls._validate_node(cls.normalize(node), allowed, "$", errors)
        return FormulaASTValidationResult(ok=(len(errors) == 0), errors=errors)

    @classmethod
    def _validate_node(cls, node: Any, allowed: Set[str], path: str, errors: List[str]) -> None:
        if not isinstance(node, dict):
            errors.append(f"{path}: AST node must be a dict, got {type(node).__name__}")
            return

        if "var" in node:
            name = str(node.get("var"))
            if not name:
                errors.append(f"{path}: var cannot be empty")
            elif allowed and name not in allowed:
                errors.append(f"{path}: variable not allowed: {name}")
            return

        if "const" in node:
            try:
                value = float(node.get("const"))
                if not math.isfinite(value):
                    errors.append(f"{path}: const must be finite")
            except Exception:
                errors.append(f"{path}: const must be numeric")
            return

        if "bool" in node:
            if not isinstance(node.get("bool"), bool):
                errors.append(f"{path}: bool leaf must be true/false")
            return

        op = str(node.get("op", "")).lower()
        if not op:
            errors.append(f"{path}: operator node missing op")
            return

        allowed_ops = cls.NUMERIC_NARY | cls.NUMERIC_BINARY | cls.NUMERIC_UNARY | cls.BOOL_NARY | cls.BOOL_UNARY | cls.COMPARE | cls.SPECIAL
        if op not in allowed_ops:
            errors.append(f"{path}: unsupported op {op!r}")
            return

        if op in cls.NUMERIC_NARY | cls.BOOL_NARY:
            args = node.get("args")
            if not isinstance(args, list) or len(args) == 0:
                errors.append(f"{path}: op {op!r} requires non-empty args list")
                return
            for i, child in enumerate(args):
                cls._validate_node(child, allowed, f"{path}.args[{i}]", errors)
            return

        if op in cls.NUMERIC_BINARY | cls.COMPARE:
            if "left" not in node or "right" not in node:
                errors.append(f"{path}: op {op!r} requires left and right")
                return
            cls._validate_node(node["left"], allowed, f"{path}.left", errors)
            cls._validate_node(node["right"], allowed, f"{path}.right", errors)
            return

        if op in cls.NUMERIC_UNARY | cls.BOOL_UNARY:
            if "arg" not in node:
                errors.append(f"{path}: op {op!r} requires arg")
                return
            cls._validate_node(node["arg"], allowed, f"{path}.arg", errors)
            return

        if op == "clip":
            args = node.get("args")
            if not isinstance(args, list) or len(args) != 3:
                errors.append(f"{path}: clip requires args=[value, low, high]")
                return
            for i, child in enumerate(args):
                cls._validate_node(child, allowed, f"{path}.args[{i}]", errors)
            return

    @classmethod
    def eval(cls, node: Any, variables: Dict[str, Any]) -> Any:
        node = cls.normalize(node)

        if not isinstance(node, dict):
            raise FormulaASTError(f"AST node must be dict, got {type(node).__name__}")

        if "var" in node:
            name = str(node["var"])
            if name not in variables:
                raise FormulaASTError(f"Variable not available: {name}")
            return variables[name]

        if "const" in node:
            return float(node["const"])

        if "bool" in node:
            return bool(node["bool"])

        op = str(node.get("op", "")).lower()

        if op == "add":
            return sum(float(cls.eval(x, variables)) for x in node["args"])
        if op == "mul":
            value = 1.0
            for child in node["args"]:
                value *= float(cls.eval(child, variables))
            return value
        if op == "sub":
            return float(cls.eval(node["left"], variables)) - float(cls.eval(node["right"], variables))
        if op == "div":
            denom = float(cls.eval(node["right"], variables))
            if abs(denom) < 1e-12:
                return 0.0
            return float(cls.eval(node["left"], variables)) / denom
        if op == "pow":
            return float(cls.eval(node["left"], variables)) ** float(cls.eval(node["right"], variables))
        if op == "neg":
            return -float(cls.eval(node["arg"], variables))
        if op == "abs":
            return abs(float(cls.eval(node["arg"], variables)))
        if op == "sqrt":
            return math.sqrt(max(0.0, float(cls.eval(node["arg"], variables))))
        if op == "exp":
            return math.exp(max(-50.0, min(50.0, float(cls.eval(node["arg"], variables)))))
        if op == "tanh":
            return math.tanh(float(cls.eval(node["arg"], variables)))
        if op == "min":
            return min(float(cls.eval(x, variables)) for x in node["args"])
        if op == "max":
            return max(float(cls.eval(x, variables)) for x in node["args"])
        if op == "clip":
            value = float(cls.eval(node["args"][0], variables))
            low = float(cls.eval(node["args"][1], variables))
            high = float(cls.eval(node["args"][2], variables))
            if low > high:
                low, high = high, low
            return max(low, min(high, value))

        if op == "and":
            return all(bool(cls.eval(x, variables)) for x in node["args"])
        if op == "or":
            return any(bool(cls.eval(x, variables)) for x in node["args"])
        if op == "not":
            return not bool(cls.eval(node["arg"], variables))

        if op in cls.COMPARE:
            left = cls.eval(node["left"], variables)
            right = cls.eval(node["right"], variables)
            if op == "lt":
                return float(left) < float(right)
            if op == "le":
                return float(left) <= float(right)
            if op == "gt":
                return float(left) > float(right)
            if op == "ge":
                return float(left) >= float(right)
            if op == "eq":
                return left == right
            if op == "ne":
                return left != right

        raise FormulaASTError(f"Unsupported AST op: {op}")

    @classmethod
    def to_expr(cls, node: Any) -> str:
        node = cls.normalize(node)

        if "var" in node:
            return str(node["var"])
        if "const" in node:
            return repr(float(node["const"]))
        if "bool" in node:
            return "True" if bool(node["bool"]) else "False"

        op = str(node.get("op", "")).lower()

        if op == "add":
            return "(" + " + ".join(cls.to_expr(x) for x in node["args"]) + ")"
        if op == "mul":
            return "(" + " * ".join(cls.to_expr(x) for x in node["args"]) + ")"
        if op == "sub":
            return f"({cls.to_expr(node['left'])} - {cls.to_expr(node['right'])})"
        if op == "div":
            return f"({cls.to_expr(node['left'])} / {cls.to_expr(node['right'])})"
        if op == "pow":
            return f"({cls.to_expr(node['left'])} ** {cls.to_expr(node['right'])})"
        if op == "neg":
            return f"(-{cls.to_expr(node['arg'])})"
        if op in {"abs", "sqrt", "exp", "tanh"}:
            return f"{op}({cls.to_expr(node['arg'])})"
        if op in {"min", "max"}:
            return f"{op}(" + ", ".join(cls.to_expr(x) for x in node["args"]) + ")"
        if op == "clip":
            return "clip(" + ", ".join(cls.to_expr(x) for x in node["args"]) + ")"
        if op == "and":
            return "(" + " and ".join(cls.to_expr(x) for x in node["args"]) + ")"
        if op == "or":
            return "(" + " or ".join(cls.to_expr(x) for x in node["args"]) + ")"
        if op == "not":
            return f"(not {cls.to_expr(node['arg'])})"

        cmp_map = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
        if op in cmp_map:
            return f"({cls.to_expr(node['left'])} {cmp_map[op]} {cls.to_expr(node['right'])})"

        raise FormulaASTError(f"Unsupported AST op: {op}")


def validate_formula_ast(node: Any, allowed_variables: Iterable[str]) -> FormulaASTValidationResult:
    return FormulaAST.validate(node, allowed_variables)


def eval_formula_ast(node: Any, variables: Dict[str, Any]) -> Any:
    return FormulaAST.eval(node, variables)


def formula_ast_to_expr(node: Any) -> str:
    return FormulaAST.to_expr(node)
