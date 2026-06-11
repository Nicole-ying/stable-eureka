from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable, List, Set


@dataclass
class FormulaValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class FormulaValidator:
    """Validate LLM-generated reward formulas using a strict AST whitelist."""

    ALLOWED_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
    }

    FORBIDDEN_NAMES = {
        "__builtins__",
        "eval",
        "exec",
        "open",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "import",
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "pickle",
        "input",
    }

    @classmethod
    def validate_expression(
        cls,
        expr: str,
        allowed_variables: Iterable[str],
        allowed_functions: Iterable[str],
    ) -> FormulaValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(expr, str) or not expr.strip():
            return FormulaValidationResult(ok=False, errors=["Expression must be a non-empty string."])

        allowed_vars: Set[str] = set(allowed_variables or [])
        allowed_funcs: Set[str] = set(allowed_functions or [])

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            return FormulaValidationResult(ok=False, errors=[f"SyntaxError: {exc}"])

        for node in ast.walk(tree):
            if type(node) not in cls.ALLOWED_NODES:
                errors.append(f"Disallowed AST node: {type(node).__name__}")

            if isinstance(node, ast.Name):
                name = node.id
                if name in cls.FORBIDDEN_NAMES:
                    errors.append(f"Forbidden name: {name}")
                elif name not in allowed_vars and name not in allowed_funcs and name not in {"True", "False"}:
                    errors.append(f"Name not allowed: {name}")

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    errors.append("Only direct calls to whitelisted functions are allowed.")
                else:
                    func_name = node.func.id
                    if func_name not in allowed_funcs:
                        errors.append(f"Function not allowed: {func_name}")

        return FormulaValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)
