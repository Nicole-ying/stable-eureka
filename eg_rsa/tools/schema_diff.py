from __future__ import annotations

from typing import Any, Dict, List


def _as_dict(schema: Any) -> Dict[str, Any]:
    if hasattr(schema, "to_dict"):
        return schema.to_dict()
    return schema or {}


class SchemaDiffTool:
    """Small schema diff helper for v1 agent tools."""

    @staticmethod
    def diff(before: Any, after: Any) -> Dict[str, Any]:
        b = _as_dict(before)
        a = _as_dict(after)
        bc = {c.get("name"): c for c in b.get("components", [])}
        ac = {c.get("name"): c for c in a.get("components", [])}
        br = {r.get("name"): r for r in b.get("event_rules", [])}
        ar = {r.get("name"): r for r in a.get("event_rules