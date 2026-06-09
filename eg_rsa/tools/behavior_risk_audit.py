from __future__ import annotations

from typing import Any, Dict, List, Set


class BehaviorRiskAuditTool:
    """Generic semantic-role + attribution risk audit.

    This tool is environment-portable:
    - it uses semantic_role metadata when available;
    - name hints are only fallback compatibility;
    - it combines role-level edit deltas, behavior evidence, retrieved lessons,
      and reward attribution to detect generic reward-search risks.

    It does not decide final reward edits. It produces structured evidence for
    RepairAgent.
    """

    name = "behavior_risk_audit"

    ROLE_TERMINAL = "terminal_success"
    ROLE_GUIDANCE = "dense_guidance"
    ROLE_STABILITY = "stability_quality"
    ROLE_CONTROL_COST = "control_cost"
    ROLE_SAFETY = "safety_constraint"

    DENSE_ROLES = {ROLE_GUIDANCE, ROLE_STABILITY, ROLE_CONTROL_COST}

    FALLBACK_HINTS = {
        ROLE_TERMINAL: ("stable_landing", "safe_contact", "landing_once", "contact_once", "terminal", "success"),
        ROLE_GUIDANCE: ("approach", "region", "progress", "forward", "distance"),
        ROLE_STABILITY: ("stability", "quality", "attitude", "angle", "velocity", "torso", "balance"),
        ROLE_CONTROL_COST: ("energy", "fuel", "action", "control", "torque", "effort"),
        ROLE_SAFETY: ("fall", "collision", "unsafe", "crash", "constraint"),
    }

    @classmethod
    def audit(
        cls,
        edit_plan: List[Dict[str, Any]],
        semantic_outcome: Dict[str, Any],
        trajectory_inspection: Dict[str, Any],
        diagnostic_report: Dict[str, Any],
        retrieved_lessons: List[Dict[str, Any]] | None = None,
        config: Dict[str, Any] | None = None,
        schema: Any | None = None,
        raw_attribution: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        config = config or {}
        retrieved_lessons = retrieved_lessons or []
        raw_attribution = raw_attribution or {}

        schema_dict = cls._to_dict(schema)
        role_map = cls._build_role_map(schema_dict, config)
        role_delta = cls._role_delta(edit_plan, role_map)
        evidence = cls._behavior_evidence(semantic_outcome, trajectory_inspection, diagnostic_report)
        attribution = cls._attribution_evidence(raw_attribution, role_map)

        risks: List[Dict[str, Any]] = []
        risks.extend(cls._risk_terminal_before_stability(role_delta, evidence, config))
        risks.extend(cls._risk_guidance_terminal_tradeoff(role_delta, evidence, config))
        risks.extend(cls._risk_control_cost_overpressure(role_delta, evidence, config))
        risks.extend(cls._risk_stability_removed_during_terminal_push(role_delta, evidence, config))
        risks.extend(cls._risk_dense_role_dominance_transfer(role_delta, evidence, attribution, config))
        risks.extend(cls._risk_memory_overlap(role_delta, retrieved_lessons))

        thresholds = cls._thresholds(config)
        max_severity = cls._max_severity(risks)
        medium_count = sum(1 for r in risks if r.get("severity") == "medium")
        high_count = sum(1 for r in risks if r.get("severity") == "high")
        weak_success = evidence["success_evidence"] < thresholds["weak_success_evidence"]

        block_medium = bool(config.get("block_medium", False))
        medium_budget = int(config.get("medium_risk_budget_when_weak_success", 0))

        audit_pass = True
        if high_count > 0:
            audit_pass = False
        elif block_medium and medium_count > 0:
            audit_pass = False
        elif weak_success and medium_count > medium_budget:
            audit_pass = False

        return {
            "tool": cls.name,
            "audit_pass": bool(audit_pass),
            "risk_basis": "semantic_role_attribution",
            "max_severity": max_severity,
            "risk_counts": {
                "high": high_count,
                "medium": medium_count,
                "low": sum(1 for r in risks if r.get("severity") == "low"),
            },
            "role_delta": role_delta,
            "behavior_evidence": evidence,
            "attribution_evidence": attribution,
            "thresholds": thresholds,
            "risks": risks,
            "repair_summary": cls._repair_summary(risks, weak_success, medium_budget),
        }

    # ============================================================
    # Schema and role helpers
    # ============================================================
    @classmethod
    def _to_dict(cls, schema: Any | None) -> Dict[str, Any]:
        if schema is None:
            return {}
        if isinstance(schema, dict):
            return schema
        if hasattr(schema, "to_dict"):
            return schema.to_dict()
        return {}

    @classmethod
    def _build_role_map(cls, schema: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Set[str]]:
        role_map: Dict[str, Set[str]] = {
            cls.ROLE_TERMINAL: set(),
            cls.ROLE_GUIDANCE: set(),
            cls.ROLE_STABILITY: set(),
            cls.ROLE_CONTROL_COST: set(),
            cls.ROLE_SAFETY: set(),
        }

        configured = config.get("reward_semantic_roles", {}) or {}
        for role, names in configured.items():
            role_map.setdefault(str(role), set()).update(str(x) for x in (names or []))

        items = list(schema.get("components", []) or []) + list(schema.get("event_rules", []) or [])
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            role = item.get("semantic_role") or item.get("metadata", {}).get("semantic_role")
            if name and role:
                role_map.setdefault(str(role), set()).add(name)

        explicit_names = set().union(*role_map.values()) if role_map else set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if not name or name in explicit_names:
                continue
            lname = name.lower()
            for role, hints in cls.FALLBACK_HINTS.items():
                if any(h in lname for h in hints):
                    role_map.setdefault(role, set()).add(name)
                    break

        return role_map

    @classmethod
    def _role_of_target(cls, target: str, role_map: Dict[str, Set[str]]) -> str | None:
        for role, names in role_map.items():
            if target in names:
                return role
        lname = target.lower()
        for role, hints in cls.FALLBACK_HINTS.items():
            if any(h in lname for h in hints):
                return role
        return None

    @classmethod
    def _role_delta(cls, edit_plan: List[Dict[str, Any]], role_map: Dict[str, Set[str]]) -> Dict[str, Any]:
        roles = [cls.ROLE_TERMINAL, cls.ROLE_GUIDANCE, cls.ROLE_STABILITY, cls.ROLE_CONTROL_COST, cls.ROLE_SAFETY]
        result: Dict[str, Any] = {
            role: {
                "touched": False,
                "max_increase_factor": 1.0,
                "min_decrease_factor": 1.0,
                "added_dense_weight": 0.0,
                "targets": [],
                "operators": [],
            }
            for role in roles
        }

        for edit in edit_plan or []:
            if not isinstance(edit, dict):
                continue
            op = str(edit.get("operator") or edit.get("op") or "")
            target = str(edit.get("target") or "")

            role = None
            if op == "add_component":
                comp = edit.get("component", {}) or {}
                target = str(comp.get("name") or target)
                role = comp.get("semantic_role") or comp.get("metadata", {}).get("semantic_role")
            elif op == "add_event_rule":
                rule = edit.get("event_rule", {}) or {}
                target = str(rule.get("name") or target)
                role = rule.get("semantic_role") or rule.get("metadata", {}).get("semantic_role")

            if role is None:
                role = cls._role_of_target(target, role_map)
            if role is None:
                continue

            slot = result.setdefault(
                str(role),
                {
                    "touched": False,
                    "max_increase_factor": 1.0,
                    "min_decrease_factor": 1.0,
                    "added_dense_weight": 0.0,
                    "targets": [],
                    "operators": [],
                },
            )
            slot["touched"] = True
            slot["targets"].append(target)
            slot["operators"].append(op)

            if op == "increase_weight":
                factor = float(edit.get("factor", edit.get("value", 1.0)) or 1.0)
                slot["max_increase_factor"] = max(float(slot["max_increase_factor"]), factor)
            elif op == "decrease_weight":
                factor = float(edit.get("factor", edit.get("value", 1.0)) or 1.0)
                slot["min_decrease_factor"] = min(float(slot["min_decrease_factor"]), factor)
            elif op == "disable_component":
                slot["min_decrease_factor"] = 0.0
            elif op == "add_component":
                comp = edit.get("component", {}) or {}
                weight = abs(float(comp.get("weight", edit.get("weight", 0.0)) or 0.0))
                slot["added_dense_weight"] = max(float(slot["added_dense_weight"]), weight)
            elif op == "add_event_rule":
                rule = edit.get("event_rule", {}) or {}
                weight = abs(float(rule.get("weight", edit.get("weight", 1.0)) or 1.0))
                slot["max_increase_factor"] = max(float(slot["max_increase_factor"]), weight)

        return result

    # ============================================================
    # Evidence extraction
    # ============================================================
    @staticmethod
    def _first_number(data: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in data:
                try:
                    return float(data.get(key, default) or default)
                except (TypeError, ValueError):
                    continue
        return float(default)

    @classmethod
    def _behavior_evidence(
        cls,
        semantic_outcome: Dict[str, Any],
        trajectory_inspection: Dict[str, Any],
        diagnostic_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnostics = diagnostic_report.get("diagnostics", {}) or {}
        flags = diagnostics.get("hack_flags", {}) or {}

        return {
            "success_evidence": cls._first_number(
                semantic_outcome,
                ["success_episode_rate", "terminal_reward_paid_episode_rate"],
                0.0,
            ),
            "stability_evidence": cls._first_number(
                semantic_outcome,
                ["stable_landing_episode_rate", "safe_contact_episode_rate", "stability_episode_rate"],
                0.0,
            ),
            "progress_evidence": cls._first_number(
                semantic_outcome,
                ["progress_score_mean", "semantic_score"],
                cls._first_number(trajectory_inspection, ["success_rate"], 0.0),
            ),
            "instability_signal": cls._first_number(
                trajectory_inspection,
                ["contact_toggle_mean", "instability_signal", "fall_rate", "stumble_rate"],
                cls._first_number(semantic_outcome, ["contact_toggle_mean", "instability_signal"], 0.0),
            ),
            "hack_score": float(diagnostics.get("hack_score", 0.0) or 0.0),
            "reward_repetition_risk": bool(semantic_outcome.get("reward_repetition_risk", False)),
            "high_reward_low_progress": bool(flags.get("high_reward_low_progress", False)),
            "shaping_goal_mismatch": bool(flags.get("shaping_goal_mismatch", False)),
        }

    @classmethod
    def _attribution_evidence(cls, raw_attribution: Dict[str, Any], role_map: Dict[str, Set[str]]) -> Dict[str, Any]:
        dominant_component = raw_attribution.get("dominant_component")
        dominant_role = cls._role_of_target(str(dominant_component or ""), role_map) if dominant_component else None
        component_stats = raw_attribution.get("component_stats", {}) or raw_attribution.get("components", {}) or {}

        role_ratio: Dict[str, float] = {}
        for comp_name, stats in component_stats.items():
            role = cls._role_of_target(str(comp_name), role_map)
            if role is None:
                continue
            ratio = 0.0
            if isinstance(stats, dict):
                ratio = float(
                    stats.get("ratio", stats.get("abs_ratio", stats.get("dominance_ratio", 0.0))) or 0.0
                )
            role_ratio[role] = role_ratio.get(role, 0.0) + ratio

        return {
            "dominant_component": dominant_component,
            "dominant_component_ratio": float(raw_attribution.get("dominant_component_ratio", 0.0) or 0.0),
            "dominant_role": dominant_role,
            "role_ratio": role_ratio,
        }

    @staticmethod
    def _thresholds(config: Dict[str, Any]) -> Dict[str, float]:
        return {
            "weak_success_evidence": float(config.get("weak_success_evidence", 0.5)),
            "weak_stability_evidence": float(config.get("weak_stability_evidence", 0.5)),
            "large_role_increase_factor": float(config.get("large_role_increase_factor", 2.5)),
            "moderate_role_increase_factor": float(config.get("moderate_role_increase_factor", 1.5)),
            "large_guidance_drop_factor": float(config.get("large_guidance_drop_factor", 0.25)),
            "moderate_guidance_drop_factor": float(config.get("moderate_guidance_drop_factor", 0.5)),
            "large_stability_drop_factor": float(config.get("large_stability_drop_factor", 0.5)),
            "dominance_ratio_threshold": float(config.get("dominance_ratio_threshold", 0.65)),
            "high_instability_signal": float(config.get("high_instability_signal", 50.0)),
        }

    # ============================================================
    # Risk templates
    # ============================================================
    @classmethod
    def _risk_terminal_before_stability(cls, role_delta: Dict[str, Any], evidence: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        th = cls._thresholds(config)
        terminal = role_delta.get(cls.ROLE_TERMINAL, {})
        stability = role_delta.get(cls.ROLE_STABILITY, {})
        guidance = role_delta.get(cls.ROLE_GUIDANCE, {})

        terminal_factor = float(terminal.get("max_increase_factor", 1.0) or 1.0)
        stability_factor = float(stability.get("max_increase_factor", 1.0) or 1.0)
        guidance_factor = float(guidance.get("min_decrease_factor", 1.0) or 1.0)

        if not terminal.get("touched", False):
            return []
        if terminal_factor <= th["large_role_increase_factor"]:
            return []
        if evidence["success_evidence"] >= th["weak_success_evidence"]:
            return []
        if evidence["stability_evidence"] >= th["weak_stability_evidence"]:
            return []

        severity = "medium" if stability_factor > 1.0 and guidance_factor > th["moderate_guidance_drop_factor"] else "high"
        return [
            cls._risk(
                "terminal_before_stability",
                severity,
                "terminal_success is increased strongly while success/stability evidence is weak.",
                {
                    "terminal_success_factor": terminal_factor,
                    "stability_quality_factor": stability_factor,
                    "dense_guidance_factor": guidance_factor,
                    "success_evidence": evidence["success_evidence"],
                    "stability_evidence": evidence["stability_evidence"],
                },
                "Reduce terminal_success change or preserve/strengthen dense_guidance and stability_quality.",
            )
        ]

    @classmethod
    def _risk_guidance_terminal_tradeoff(cls, role_delta: Dict[str, Any], evidence: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        th = cls._thresholds(config)
        guidance = role_delta.get(cls.ROLE_GUIDANCE, {})
        terminal = role_delta.get(cls.ROLE_TERMINAL, {})
        guidance_factor = float(guidance.get("min_decrease_factor", 1.0) or 1.0)
        terminal_factor = float(terminal.get("max_increase_factor", 1.0) or 1.0)

        if evidence["success_evidence"] >= th["weak_success_evidence"]:
            return []
        if guidance_factor > th["moderate_guidance_drop_factor"]:
            return []
        if terminal_factor <= th["moderate_role_increase_factor"]:
            return []

        severity = "high" if (
            guidance_factor <= th["large_guidance_drop_factor"]
            and terminal_factor > th["large_role_increase_factor"]
        ) else "medium"

        return [
            cls._risk(
                "guidance_terminal_tradeoff_under_weak_success",
                severity,
                "dense_guidance is reduced while terminal_success pressure is increased before success is reliable.",
                {
                    "dense_guidance_factor": guidance_factor,
                    "terminal_success_factor": terminal_factor,
                    "success_evidence": evidence["success_evidence"],
                },
                "Use a milder dense_guidance reduction or keep guidance until success evidence improves.",
            )
        ]

    @classmethod
    def _risk_control_cost_overpressure(cls, role_delta: Dict[str, Any], evidence: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        th = cls._thresholds(config)
        control = role_delta.get(cls.ROLE_CONTROL_COST, {})
        terminal = role_delta.get(cls.ROLE_TERMINAL, {})

        control_factor = float(control.get("max_increase_factor", 1.0) or 1.0)
        terminal_factor = float(terminal.get("max_increase_factor", 1.0) or 1.0)

        if control_factor <= th["large_role_increase_factor"]:
            return []
        if evidence["success_evidence"] >= th["weak_success_evidence"]:
            return []

        severity = "high" if terminal_factor > th["large_role_increase_factor"] else "medium"
        return [
            cls._risk(
                "control_cost_overpressure_under_weak_success",
                severity,
                "control_cost is increased strongly while success is weak; this may suppress corrective actions or exploration.",
                {
                    "control_cost_factor": control_factor,
                    "terminal_success_factor": terminal_factor,
                    "success_evidence": evidence["success_evidence"],
                },
                "Use smaller control_cost changes until task behavior is stable.",
            )
        ]

    @classmethod
    def _risk_stability_removed_during_terminal_push(cls, role_delta: Dict[str, Any], evidence: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        th = cls._thresholds(config)
        stability = role_delta.get(cls.ROLE_STABILITY, {})
        terminal = role_delta.get(cls.ROLE_TERMINAL, {})

        stability_factor = float(stability.get("min_decrease_factor", 1.0) or 1.0)
        terminal_factor = float(terminal.get("max_increase_factor", 1.0) or 1.0)

        if stability_factor > th["large_stability_drop_factor"]:
            return []
        if terminal_factor <= 1.0:
            return []

        return [
            cls._risk(
                "stability_removed_during_terminal_push",
                "medium",
                "stability_quality is reduced while terminal_success pressure is increased.",
                {
                    "stability_quality_factor": stability_factor,
                    "terminal_success_factor": terminal_factor,
                },
                "Do not reduce stability_quality when increasing terminal_success.",
            )
        ]

    @classmethod
    def _risk_dense_role_dominance_transfer(
        cls,
        role_delta: Dict[str, Any],
        evidence: Dict[str, Any],
        attribution: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        th = cls._thresholds(config)
        risks: List[Dict[str, Any]] = []

        if evidence["success_evidence"] >= th["weak_success_evidence"]:
            return risks

        dominant_role = attribution.get("dominant_role")
        dominant_ratio = float(attribution.get("dominant_component_ratio", 0.0) or 0.0)

        for role in cls.DENSE_ROLES:
            delta = role_delta.get(role, {})
            increased = float(delta.get("max_increase_factor", 1.0) or 1.0) > 1.0 or float(delta.get("added_dense_weight", 0.0) or 0.0) > 0.0
            if not increased:
                continue

            # Current dominant role is already dense and this edit strengthens it.
            if dominant_role == role and dominant_ratio >= th["dominance_ratio_threshold"]:
                risks.append(
                    cls._risk(
                        "dense_role_dominance_amplification",
                        "high",
                        "A dense semantic role already dominates reward attribution and the edit strengthens the same role.",
                        {
                            "role": role,
                            "dominant_component": attribution.get("dominant_component"),
                            "dominant_component_ratio": dominant_ratio,
                            "role_increase_factor": delta.get("max_increase_factor", 1.0),
                        },
                        "Do not strengthen a dense role that already dominates while success is weak; reduce or diversify reward pressure.",
                    )
                )

            # A different dense role is increased under weak success; this can transfer exploitation.
            elif dominant_ratio >= th["dominance_ratio_threshold"] and dominant_role in cls.DENSE_ROLES:
                risks.append(
                    cls._risk(
                        "dense_role_dominance_transfer",
                        "medium",
                        "Reward attribution is already dense-role dominated, and the edit strengthens another dense role under weak success.",
                        {
                            "current_dominant_role": dominant_role,
                            "new_strengthened_role": role,
                            "dominant_component": attribution.get("dominant_component"),
                            "dominant_component_ratio": dominant_ratio,
                        },
                        "Avoid moving exploitation from one dense role to another; prefer terminal repair only with stable guidance and small factors.",
                    )
                )

        return risks

    @classmethod
    def _risk_memory_overlap(cls, role_delta: Dict[str, Any], lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        touched_roles = {
            role for role, info in role_delta.items()
            if isinstance(info, dict) and bool(info.get("touched", False))
        }
        risks: List[Dict[str, Any]] = []

        for lesson in lessons or []:
            lesson_type = str(lesson.get("lesson_type", "")).lower()
            if "regression" not in lesson_type and "dominance" not in lesson_type:
                continue

            applicability = lesson.get("applicability", {}) or {}
            lesson_roles = set(applicability.get("semantic_roles_touched", []) or [])
            if not lesson_roles:
                for edit in lesson.get("edit_plan", []) or []:
                    if not isinstance(edit, dict):
                        continue
                    target = str(edit.get("target", "")).lower()
                    for role, hints in cls.FALLBACK_HINTS.items():
                        if any(h in target for h in hints):
                            lesson_roles.add(role)

            overlap = touched_roles & lesson_roles
            if overlap:
                risks.append(
                    cls._risk(
                        "similar_role_pattern_to_regression_lesson",
                        "medium",
                        "The proposed edit touches semantic roles associated with a retrieved regression lesson.",
                        {
                            "overlap_roles": sorted(overlap),
                            "lesson_type": lesson_type,
                        },
                        "RepairAgent should explain why this role-level pattern will not repeat the prior regression.",
                    )
                )

        return risks

    @staticmethod
    def _risk(risk_type: str, severity: str, reason: str, evidence: Dict[str, Any], repair_instruction: str) -> Dict[str, Any]:
        return {
            "risk_type": risk_type,
            "severity": severity,
            "reason": reason,
            "role_level_evidence": evidence,
            "repair_instruction": repair_instruction,
        }

    @staticmethod
    def _max_severity(risks: List[Dict[str, Any]]) -> str:
        rank = {"low": 0, "medium": 1, "high": 2}
        out = "low"
        for risk in risks:
            sev = str(risk.get("severity", "low"))
            if rank.get(sev, 0) > rank.get(out, 0):
                out = sev
        return out

    @staticmethod
    def _repair_summary(risks: List[Dict[str, Any]], weak_success: bool, medium_budget: int) -> str:
        if not risks:
            return "No role-level attribution risk detected."
        if any(r.get("severity") == "high" for r in risks):
            return "High role-level attribution risk detected; ask RepairAgent to reduce aggressive dense/terminal role changes."
        if weak_success and sum(1 for r in risks if r.get("severity") == "medium") > medium_budget:
            return "Multiple medium risks under weak success; repair or continue_training is safer than direct execution."
        return "Medium role-level risk detected; ask RepairAgent to justify or soften the edit."
