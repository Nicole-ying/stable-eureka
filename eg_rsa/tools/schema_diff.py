from __future__ import annotations

import json
from typing import Any, Dict, List


def _to_dict(schema: Any) -> Dict[str, Any]:
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "to_dict"):
        return schema.to_dict()
    raise TypeError(f"Unsupported schema type: {type(schema)!r}")


def _items(data: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    raw = data.get(key, {})
    if isinstance(raw, dict):
        return {str(k): dict(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("id") or item.get("component")
            if name:
                out[str(name)] = dict(item)
        return out
    return {}


def _changed(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name in sorted(set(before) & set(after)):
        b = before[name]
        a = after[name]
        if json.dumps(b, sort_keys=True) == json.dumps(a, sort_keys=True):
            continue
        changed_fields = []
        for key in sorted(set(b) | set(a)):
            if b.get(key) != a.get(key):
                changed_fields.append({"field": key, "before": b.get(key), "after": a.get(key)})
        rows.append({"name": name, "changed_fields": changed_fields})
    return rows


def diff_schemas(before_schema: Any, after_schema: Any) -> Dict[str, Any]:
    before = _to_dict(before_schema)
    after = _to_dict(after_schema)

    b_comp = _items(before, "components")
    a_comp = _items(after, "components")
    b_rules = _items(before, "event_rules")
    a_rules = _items(after, "event_rules")

    return {
        "components": {
            "added": sorted(set(a_comp) - set(b_comp)),
            "removed": sorted(set(b_comp) - set(a_comp)),
            "changed": _changed(b_comp, a_comp),
        },
        "event_rules": {
            "added": sorted(set(a_rules) - set(b_rules)),
            "removed": sorted(set(b_rules) - set(a_rules)),
            "changed": _changed(b_rules, a_rules),
        },
        "summary": {
            "num_component_changes": len(set(a_comp) ^ set(b_comp)) + len(_changed(b_comp, a_comp)),
            "num_event_rule_changes": len(set(a_rules) ^ set(b_rules)) + len(_changed(b_rules, a_rules)),
        },
    }


class SchemaDiffTool:
    name = "schema_diff"

    @staticmethod
    def run(before_schema: Any, after_schema: Any) -> Dict[str, Any]:
        return diff_schemas(before_schema, after_schema)
