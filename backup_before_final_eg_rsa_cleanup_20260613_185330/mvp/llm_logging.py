from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    # Conservative language-agnostic heuristic.
    return max(1, int(len(text) / 3.5))


def write_llm_call(
    log_dir: Path,
    system: str,
    user: str,
    response: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)

    system_path = log_dir / "system.txt"
    user_path = log_dir / "user.txt"
    response_path = log_dir / "response.txt"
    budget_path = log_dir / "budget.json"

    system_path.write_text(system, encoding="utf-8")
    user_path.write_text(user, encoding="utf-8")
    response_path.write_text(response, encoding="utf-8")

    budget = {
        "system_chars": len(system),
        "user_chars": len(user),
        "response_chars": len(response),
        "estimated_system_tokens": estimate_tokens(system),
        "estimated_user_tokens": estimate_tokens(user),
        "estimated_input_tokens": estimate_tokens(system) + estimate_tokens(user),
        "estimated_output_tokens": estimate_tokens(response),
        "paths": {
            "system": str(system_path),
            "user": str(user_path),
            "response": str(response_path),
        },
        "metadata": metadata or {},
    }
    budget_path.write_text(json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")
    return budget
