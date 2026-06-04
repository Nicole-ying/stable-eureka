from __future__ import annotations

import ast
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.ClassDef,
    ast.Lambda,
    ast.Try,
    ast.Raise,
)

SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "float": float,
    "int": int,
    "bool": bool,
    "range": range,
    "enumerate": enumerate,
    "hasattr": hasattr,
}


def validate_reward_code(code: str) -> None:
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "compute_reward":
        raise ValueError("Reward code must define exactly one function named compute_reward.")

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"Forbidden syntax in reward code: {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "eval", "open", "__import__", "compile", "input"}:
                raise ValueError(f"Forbidden function call: {node.func.id}")


@dataclass
class RewardProgram:
    code: str
    reward_clip: Optional[float] = 10.0
    error_fallback: str = "original"
    error_count: int = field(default=0, init=False)
    last_error: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        validate_reward_code(self.code)
        namespace: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "math": math,
        }
        if np is not None:
            namespace["np"] = np
        exec(compile(self.code, "<llm_reward>", "exec"), namespace)
        self._fn: Callable[..., Any] = namespace["compute_reward"]

    def __call__(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        original_reward: float,
        info: Dict[str, Any],
        training_progress: float = 0.0,
    ) -> float:
        try:
            value = self._fn(obs, action, next_obs, original_reward, info, training_progress)
            reward = float(value)
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            reward = self._fallback_reward(original_reward)
        if not math.isfinite(reward):
            self.error_count += 1
            self.last_error = "non-finite reward"
            reward = self._fallback_reward(original_reward)
        if self.reward_clip is not None:
            limit = abs(float(self.reward_clip))
            reward = max(-limit, min(limit, reward))
        return reward

    def _fallback_reward(self, original_reward: float) -> float:
        if self.error_fallback == "zero":
            return 0.0
        return float(original_reward)
