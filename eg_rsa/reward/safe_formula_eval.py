from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np

from eg_rsa.reward.formula_validator import FormulaValidator


def _clip_value(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), float(low), float(high)))


ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "tanh": math.tanh,
    "clip": _clip_value,
}


def safe_eval_formula(
    expr: str,
    variables: Dict[str, Any],
    allowed_functions: Dict[str, Callable[..., Any]] | None = None,
) -> float:
    funcs = allowed_functions or ALLOWED_FUNCTIONS
    validation = FormulaValidator.validate_expression(
        expr,
        allowed_variables=set(variables.keys()),
        allowed_functions=set(funcs.keys()),
    )
    if not validation.ok:
        raise ValueError(f"Unsafe formula expression: {expr}; errors={validation.errors}")

    safe_locals: Dict[str, Any] = {}
    safe_locals.update(funcs)
    safe_locals.update(variables)

    value = eval(compile(expr, "<eg_rsa_formula>", "eval"), {"__builtins__": {}}, safe_locals)
    return float(value)
