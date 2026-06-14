from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from .json_tools import write_json, write_text


FORBIDDEN_TOKENS = [
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "input(",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "shutil",
    "pickle",
    "os.",
    "sys.",
]

ALLOWED_IMPORTS = {"math", "numpy", "np"}


def validate_reward_code(reward_code: str) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if "def compute_reward(self, state, m_power, s_power, terminated)" not in reward_code:
        errors.append("Reward code must define: def compute_reward(self, state, m_power, s_power, terminated)")

    for token in FORBIDDEN_TOKENS:
        if token in reward_code:
            errors.append(f"Forbidden token detected: {token}")

    try:
        tree = ast.parse(reward_code)
    except SyntaxError as exc:
        errors.append(f"SyntaxError: {exc}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    fn_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_reward"]
    if len(fn_nodes) != 1:
        errors.append("Exactly one top-level compute_reward function is required")
    else:
        args = [arg.arg for arg in fn_nodes[0].args.args]
        expected = ["self", "state", "m_power", "s_power", "terminated"]
        if args[:5] != expected:
            errors.append(f"compute_reward args must start with {expected}, got {args}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [(node.module or "").split(".")[0]]
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    errors.append(f"Import not allowed in reward code: {name}")

    if "individual_reward" not in reward_code:
        warnings.append("Reward code should return component diagnostics through individual_reward")
    if "return" not in reward_code:
        errors.append("Reward code must return reward and individual_reward")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def write_reward_code_files(reward_code: str, output_dir: str | Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_reward_code(reward_code)
    write_text(output_dir / "reward_code.py", reward_code)
    write_json(output_dir / "validation_report.json", validation)
    return validation


def indent_as_class_method(function_code: str) -> str:
    """Indent a top-level function so it can be appended as a class method."""
    function_code = textwrap.dedent(function_code).strip() + "\n"
    return "\n" + textwrap.indent(function_code, "    ")
