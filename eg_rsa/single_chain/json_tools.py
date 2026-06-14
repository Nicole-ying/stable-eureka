from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from an LLM response.

    Agents are prompted to return JSON only, but local models may still wrap the
    object in prose or Markdown fences. This helper keeps the pipeline tolerant
    while still failing loudly if no valid JSON object can be recovered.
    """
    stripped = _strip_code_fences(text)
    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON root must be an object")
        return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object start found in LLM response")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : idx + 1]
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM JSON root must be an object")
                return parsed

    raise ValueError("Could not recover a complete JSON object from LLM response")
