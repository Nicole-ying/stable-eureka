from __future__ import annotations

import json
import re
from typing import Any, Dict


_JSON_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response.

    The edit agent must return JSON.  This parser accepts either raw JSON or a
    fenced json code block, then fails loudly if parsing is impossible.
    """

    text = text.strip()
    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        text = block_match.group(1).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in model response: {exc}") from exc
        obj = json.loads(text[start : end + 1])

    if not isinstance(obj, dict):
        raise ValueError("The parsed JSON response must be an object")
    return obj
