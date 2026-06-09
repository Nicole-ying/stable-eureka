from __future__ import annotations

from typing import Any, Dict, List, Set


class BehaviorRiskAuditTool:
    """Generic role-based behavior risk audit.

    This tool is intentionally not environment-specific. It reasons over
    semantic roles such as terminal_success, dense_guidance, stability_quality,
    and control_cost. Component-name hints are only fallback compatibility for
    older schemas that do not yet carry semantic_role metadata.

    The tool does not decide final actions. It produces structured role-level
    risk evidence for RepairAgent.
    """

    name = "behavior_risk_audit"

    ROLE_TERMINAL = "terminal_success"
    ROLE_GUIDANCE = "dense_guidance"
    ROLE_STABILITY = "stability_quality"
    ROLE_CONTROL_COST = "control_cost"
    ROLE_SAFETY = "safety_constraint"

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
    ) -> Dict[str, Any]:
        config = config or {}
        retrieved_lessons = retrieved_lessons or []

        schema_dict = cls._to_dict(schema)
        role_map = cls._build_role_map(schema_dict, config)

        role_delta = cls._role_delta(edit_plan, role_map)
        evidence = cls._behavior_evidence(semantic_outcome, trajectory_inspection, diagnostic_report, config)

        risks: List[Dict[str, Any]] = []
        risks.extend(cls._template_terminal_before_stability(role_delta, evidence, config))
        risks.extend(cls._template_guidance_removed_too_early(role_delta, evidence, config))
        risks.extend(cls._template_control_cost_overpressure(role_delta, evidence, config))
        risks.extend(cls._template_stability_removed_during_terminal_push(role_delta, evidence, config))
        risks.extend(cls._memory_risks(role_delta, retrieved_lessons))

        severity_rank = {"low": 0, "medium": 1, "high": 2}
        max_severity = "low"
        for risk in risks:
            sev = str(risk.get("severity", "low"))
            if severity_rank.get(sev, 0) > severity_rank.get(max_severity, 0):
                max_severity = sev

        block_medium = bool(config.get("block_medium", False))
        audit_pass = not any(
            risk.get("severity") == "high" or (block_medium and risk.get("severity") == "medium")
            for risk in risks
        )

        return {
            "tool": cls.name,
            "audit_pass": bool(audit_pass),
            "max_severity": max_severity,
            "risk_basis": "semantic_role",
            "role_delta": role_delta,
            "behavior_evidence": evidence,
            "thresholds": cls._thresholds(config),
            "risks": risks,
            "repair_summary": cls._repair_summary(risks),
        }

    # ------------------------------------------------------------------
    # Role extraction
    # ------------------------------------------------------------------
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

        for item in list(schema.get("components", []) or []) + list(schema.get("event_rules", []) or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            role = item.get("semantic_role") or item.get("metadata", {}).get("semantic_role")
            if role and name:
                role_map.setdefault(str(role), set()).add(name)

        # Compatibility fallback: if no explicit role exists for a name, infer
        # weakly from name hints. This is not the main path.
        all_known_names = {
            str(item.get("name", ""))
            for item in list(schema.get("components", []) or []) + list(schema.get("event_rules", []) or [])
            if isinstance(item, dict)
        }
        explicit_names = set().union(*role_map.values()) if role_map else set()
        for name in all_known_names - explicit_names:
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
        result: Dict[str, Any] = {
            role: {
                "max_increase_factor": 1.0,
                "min_decrease_factor": 1.0,
                "added_dense_weight": 0.0,
                "touched": False,
                "targets": [],
                "operators": [],
            }
            for role in [
                cls.ROLE_TERMINAL,
                cls.ROLE_GUIDANCE,
                cls.ROLE_STABILITY,
                cls.ROLE_CONTROL_COST,
                cls.ROLE_SAFETY,
            ]
        }

        for edit in edit_plan or []:
            op = str(edit.get("operator") or edit.get("op") or "")
            target = str(edit.get("target") or "")

            role = None
            if op == "add_component":
                comp = edit.get("component", {}) or {}
                target = str(comp.get("name") or target)
                role = comp.get("semantic_role") or comp.get("metadata", {}).get("semantic_role")
                if role is None:
                    role = cls._role_of_target(target, role_map)
            elif op == "add_event_rule":
                rule = edit.get("event_rule", {}) or {}
                target = str(rule.get("name") or target)
                role = rule.get("semantic_role") or rule.get("metadata", {}).get("semantic_role")
                if role is None:
                    role = cls._role_of_target(target, role_map)
            else:
                role = cls._role_of_target(target, role_map)

            if role is None:
                continue

            slot = result.setdefault(
                str(role),
                {
                    "max_increase_factor": 1.0,
                    "min_decrease_factor": 1.0,
                    "added_dense_weight": 0.0,
                    "touched": False,
                    "targets": [],
                    "operators": [],
                },
            )
            slot["touched"] = True
            if target:
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
                factor = abs(float(rule.get("weight", edit.get("weight", 1.0)) or 1.0))
                slot["max_increase_factor"] = max(float(slot["max_increase_factor"]), factor)

        return result

    # ------------------------------------------------------------------
    # Generic behavior evidence
    # ------------------------------------------------------------------
    @classmethod
    def _behavior_evidence(
        cls,
        semantic_outcome: Dict[str, Any],
        trajectory_inspection: Dict[str, Any],
        diagnostic_report: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnostics = diagnostic_report.get("diagnostics", {}) or {}
        flags = diagnostics.get("hack_flags", {}) or {}

        success = cls._first_number(
            semantic_outcome,
            ["success_episode_rate", "terminal_reward_paid_episode_rate"],
            0.0,
        )
        stability = cls._first_number(
            semantic_outcome,
            ["stable_landing_episode_rate", "safe_contact_episode_rate", "stability_episode_rate"],
            0.0,
        )
        progress = cls._first_number(
            semantic_outcome,
            ["progress_score_mean", "semantic_score"],
            cls._first_number(trajectory_inspection, ["success_rate"], 0.0),
        )
        instability = cls._first_number(
            trajectory_inspection,
            ["contact_toggle_mean", "instability_signal", "fall_rate", "stumble_rate"],
            cls._first_number(semantic_outcome, ["contact_toggle_mean", "instability_signal"], 0.0),
        )

        return {
            "success_evidence": float(success),
            "stability_evidence": float(stability),
            "progress_evidence": float(progress),
            "instability_signal": float(instability),
            "hack_score": float(diagnostics.get("hack_score", 0.0) or 0.0),
            "reward_repetition_risk": bool(semantic_outcome.get("reward_repetition_risk", False)),
            "high_reward_low_progress": bool(flags.get("high_reward_low_progress", False)),
            "shaping_goal_mismatch": bool(flags.get("shaping_goal_mismatch", False)),
        }

    @staticmethod
    def _first_number(data: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in data:
                try:
                    return float(data.get(key, default) or default)
                except (TypeError, ValueError):
                    continue
        return float(default)

    @staticmethod
    def _thresholds(config: Dict[str, Any]) -> Dict[str, float]:
        return {
            "weak_success_evidence": float(config.get("weak_success_evidence", 0.5)),
            "weak_stability_evidence": float(config.get("weak_stability_evidence", 0.5)),
            "large_role_increase_factor": float(config.get("large_role_increase_factor", 2.5)),
            "large_guidance_drop_factor": float(config.get("large_guidance_drop_factor", 0.25)),
            "large_stability_drop_factor": float(config.get("large_stability_drop_factor", 0.5)),
            "high_instability_signal": float(config.get("high_instability_signal", 50.0)),
        }

    # ------------------------------------------------------------------
    # Generic risk templates
    # ------------------------------------------------------------------
    @classmethod
    def _template_terminal_before_stability(
        cls,
        role_delta: Dict[str, Any],
        evidence: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        th = cls._thresholds(config)
        terminal = role_delta.get(cls.ROLE_TERMINAL, {})
        stability = role_delta.get(cls.ROLE_STABILITY, {})
        terminal_factor = float(terminal.get("max_increase_factor", 1.0) or 1.0)
        stability_factor = float(stability.get("max_increase_factor", 1.0) or 1.0)

        if not terminal.get("touched", False):
            return []
        if terminal_factor <= th["large_role_increase_factor"]:
            return []
        if evidence["success_evidence"] >= th["weak_success_evidence"]:
            return []
        if evidence["stability_evidence"] >= th["weak_stability_evidence"]:
            return []

        severity = "high"
        if stability_factor > 1.0:
            severity = "medium"

        return [
            cls._risk(
                "terminal_before_stability",
                severity,
                "terminal_success is increased strongly while success/stability evidence is weak.",
                {
                    "terminal_success_factor": terminal_factor,
                    "stability_quality_factor": stability_factor,
                    "success_evidence": evidence["success_evidence"],
                    "stability_evidence": evidence["stability_evidence"],
                },
                "Reduce terminal_success increase or pair it with stronger stability_quality / dense_guidance support.",
            )
        ]

    @classmethod
    def _template_guidance_removed_too_early(
        cls,
        role_delta: Dict[str, Any],
        evidence: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        th = cls._thresholds(config)
        guidance = role_delta.get(cls.ROLE_GUIDANCE, {})
        terminal = role_delta.get(cls.ROLE_TERMINAL, {})
        guidance_factor = float(guidance.get("min_decrease_factor", 1.0) or 1.0)
        terminal_factor = float(terminal.get("max_increase_factor", 1.0) or 1.0)

        if guidance_factor > th["large_guidance_drop_factor"]:
            return []
        if evidence["success_evidence"] >= th["weak_success_evidence"]:
            return []

        return [
            cls._risk(
                "guidance_removed_too_early",
                "medium",
                "dense_guidance is reduced heavily before success evidence is reliable.",
                {
                    "dense_guidance_factor": guidance_factor,
                    "terminal_success_factor": terminal_factor,
                    "success_evidence": evidence["success_evidence"],
                },
                "Use a milder dense_guidance reduction or keep progress/navigation shaping until success is stable.",
            )
        ]

    @classmethod
    def _template_control_cost_overpressure(
        cls,
        role_delta: Dict[str, Any],
        evidence: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
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
                "control_cost_overpressure",
                severity,
                "control_cost is increased strongly while success is weak; this can suppress corrective actions or exploration.",
                {
                    "control_cost_factor": control_factor,
                    "terminal_success_factor": terminal_factor,
                    "success_evidence": evidence["success_evidence"],
                },
                "Use smaller control_cost changes until success is stable; avoid coupling large terminal and control-cost increases.",
            )
        ]

    @classmethod
    def _template_stability_removed_during_terminal_push(
        cls,
        role_delta: Dict[str, Any],
        evidence: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
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
    def _memory_risks(cls, role_delta: Dict[str, Any], lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        touched_roles = {
            role
            for role, info in role_delta.items()
            if isinstance(info, dict) and bool(info.get("touched", False))
        }
        risks: List[Dict[str, Any]] = []
        for lesson in lessons or []:
            lesson_type = str(lesson.get("lesson_type", "")).lower()
            if "regression" not in lesson_type and "dominance" not in lesson_type:
                continue

            lesson_roles = set()
            applicability = lesson.get("applicability", {}) or {}
            if isinstance(applicability, dict):
                role = applicability.get("semantic_role")
                if role:
                    lesson_roles.add(str(role))

            # Fallback: infer roles from previous edit targets if no explicit role exists.
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
    def _risk(
        risk_type: str,
        severity: str,
        reason: str,
        evidence: Dict[str, Any],
        repair_instruction: str,
    ) -> Dict[str, Any]:
        return {
            "risk_type": risk_type,
            "severity": severity,
            "reason": reason,
            "role_level_evidence": evidence,
            "repair_instruction": repair_instruction,
        }

    @staticmethod
    def _repair_summary(risks: List[Dict[str, Any]]) -> str:
        if not risks:
            return "No role-level behavior risk detected."
        if any(r.get("severity") == "high" for r in risks):
            return "High role-level behavior risk detected; ask RepairAgent to reduce aggressive role changes and preserve guidance/stability."
        return "Medium role-level behavior risk detected; ask RepairAgent to justify or soften the edit."
