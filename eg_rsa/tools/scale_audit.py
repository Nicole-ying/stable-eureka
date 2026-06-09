from __future__ import annotations

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
            if isinstance(item, dict):
                name = item.get("name") or item.get("id") or item.get("component")
                if name:
                    out[str(name)] = dict(item)
        return out
    return {}


class ScaleAuditTool:
    """Audit reward edit scale before executing risky structural edits.

    Main purpose:
    prevent a new dense reward or penalty from dominating one-time terminal
    reward, as happened with large stagnation penalties in experiments.
    """

    name = "scale_audit"

    @staticmethod
    def audit(
        schema: Any,
        edit_plan: List[Dict[str, Any]],
        trajectories: List[Dict[str, Any]] | None = None,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        config = config or {}
        data = _to_dict(schema)
        components = _items(data, "components")
        rules = _items(data, "event_rules")
        horizon = int(config.get("horizon") or ScaleAuditTool._infer_horizon(trajectories) or 1000)
        safe_ratio = float(config.get("max_dense_to_terminal_ratio", 0.25))

        terminal_scale = ScaleAuditTool._terminal_scale(rules)
        reports = []
        hard_fail = False

        for edit in edit_plan or []:
            op = edit.get("operator") or edit.get("op")
            target = edit.get("target")
            report = {
                "operator": op,
                "target": target,
                "risk_level": "low",
                "warnings": [],
                "estimated_episode_contribution": None,
                "terminal_scale": terminal_scale,
            }

            if op == "add_component":
                weight = abs(float(edit.get("component", {}).get("weight", edit.get("weight", 0.0)) or 0.0))
                clip = edit.get("component", {}).get("clip", edit.get("clip"))
                per_step = ScaleAuditTool._per_step_bound(weight, clip)
                estimated = per_step * horizon
                report["estimated_episode_contribution"] = estimated
                ratio = estimated / max(terminal_scale, 1e-6)
                report["dense_to_terminal_ratio"] = ratio
                if ratio > safe_ratio:
                    report["risk_level"] = "high"
                    report["warnings"].append(
                        f"New dense component may contribute {ratio:.2f}x terminal scale; reduce weight or run candidate sweep."
                    )
                    hard_fail = True

            elif op in {"increase_weight", "decrease_weight"} and target in components:
                old_weight = abs(float(components[target].get("weight", 0.0) or 0.0))
                factor = float(edit.get("factor", edit.get("value", 1.0)) or 1.0)
                new_weight = abs(old_weight * factor)
                report["old_weight"] = old_weight
                report["new_weight"] = new_weight
                if factor > 5.0:
                    report["risk_level"] = "medium"
                    report["warnings"].append("Large weight increase; consider candidate sweep.")

            elif op in {"add_event_rule", "add_duration_condition"}:
                report["warnings"].append("Event rule changes should be checked with trigger-rate and semantic outcome after training.")

            reports.append(report)

        return {
            "tool": ScaleAuditTool.name,
            "audit_pass": not hard_fail,
            "terminal_scale": terminal_scale,
            "horizon": horizon,
            "reports": reports,
        }

    @staticmethod
    def _terminal_scale(event_rules: Dict[str, Dict[str, Any]]) -> float:
        values = []
        for rule in event_rules.values():
            if rule.get("one_time", False) or "once" in str(rule.get("name", "")).lower():
                values.append(abs(float(rule.get("weight", 0.0) or 0.0)))
        return max(values) if values else 1.0

    @staticmethod
    def _per_step_bound(weight: float, clip: Any) -> float:
        if isinstance(clip, (list, tuple)) and len(clip) == 2:
            return abs(weight) * max(abs(float(clip[0])), abs(float(clip[1])))
        return abs(weight)

    @staticmethod
    def _infer_horizon(trajectories: List[Dict[str, Any]] | None) -> int | None:
        if not trajectories:
            return None
        lengths = []
        for traj in trajectories:
            summary = traj.get("summary", {}) if isinstance(traj, dict) else {}
            if "episode_length" in summary:
                lengths.append(int(summary["episode_length"]))
        return int(sum(lengths) / len(lengths)) if lengths else None
