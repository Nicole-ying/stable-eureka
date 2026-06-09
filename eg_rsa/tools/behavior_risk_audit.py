from __future__ import annotations

from typing import Any, Dict, List, Tuple


class BehaviorRiskAuditTool:
    """Detect behavior-level risks in a proposed edit plan.

    This tool does not make the final decision. It produces structured risk
    evidence for RepairAgent. Runner may use a high-risk report to request
    repair, not to silently rewrite the plan.
    """

    name = "behavior_risk_audit"

    TERMINAL_HINTS = (
        "stable_landing",
        "safe_contact",
        "landing_once",
        "contact_once",
        "terminal",
    )

    APPROACH_HINTS = (
        "approach",
        "region",
        "progress",
    )

    STABILITY_HINTS = (
        "stability",
        "landing_quality",
        "attitude",
        "angle",
        "velocity",
    )

    ENERGY_HINTS = (
        "energy",
        "fuel",
        "action",
        "control",
    )

    @classmethod
    def audit(
        cls,
        edit_plan: List[Dict[str, Any]],
        semantic_outcome: Dict[str, Any],
        trajectory_inspection: Dict[str, Any],
        diagnostic_report: Dict[str, Any],
        retrieved_lessons: List[Dict[str, Any]] | None = None,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        config = config or {}
        retrieved_lessons = retrieved_lessons or []

        success_rate = float(semantic_outcome.get("success_episode_rate", 0.0) or 0.0)
        stable_rate = float(semantic_outcome.get("stable_landing_episode_rate", 0.0) or 0.0)
        terminal_rate = float(semantic_outcome.get("terminal_reward_paid_episode_rate", 0.0) or 0.0)
        contact_toggle = float(
            trajectory_inspection.get(
                "contact_toggle_mean",
                semantic_outcome.get("contact_toggle_mean", 0.0),
            )
            or 0.0
        )
        hack_score = float((diagnostic_report.get("diagnostics", {}) or {}).get("hack_score", 0.0) or 0.0)

        terminal_pressure = cls._max_factor(edit_plan, cls.TERMINAL_HINTS, positive=True)
        energy_pressure = cls._max_factor(edit_plan, cls.ENERGY_HINTS, positive=True)
        approach_drop = cls._min_factor(edit_plan, cls.APPROACH_HINTS, positive=False)
        stability_drop = cls._min_factor(edit_plan, cls.STABILITY_HINTS, positive=False)

        high_contact_toggle = contact_toggle >= float(config.get("high_contact_toggle", 50.0))
        weak_success = success_rate < float(config.get("weak_success_rate", 0.5))
        weak_stability = stable_rate < float(config.get("weak_stable_rate", 0.5))
        terminal_big = terminal_pressure >= float(config.get("large_terminal_factor", 2.5))
        energy_big = energy_pressure >= float(config.get("large_energy_factor", 2.5))
        approach_big_drop = approach_drop <= float(config.get("large_approach_drop_factor", 0.25))
        stability_big_drop = stability_drop <= float(config.get("large_stability_drop_factor", 0.5))

        risks: List[Dict[str, Any]] = []

        if terminal_big and weak_stability:
            risks.append(
                cls._risk(
                    "terminal_pressure_before_stability",
                    "high",
                    "Terminal reward is increased strongly before stable landing evidence is reliable.",
                    {
                        "terminal_factor": terminal_pressure,
                        "stable_landing_episode_rate": stable_rate,
                        "success_episode_rate": success_rate,
                    },
                    "Reduce terminal multiplier or pair it with stronger stability/landing-quality guidance.",
                )
            )

        if terminal_big and energy_big and weak_success:
            risks.append(
                cls._risk(
                    "terminal_energy_coupling_instability",
                    "high",
                    "Terminal pressure and energy penalty are both increased while success is still weak; this may encourage fast unstable contact or reduce corrective control.",
                    {
                        "terminal_factor": terminal_pressure,
                        "energy_factor": energy_pressure,
                        "success_episode_rate": success_rate,
                    },
                    "Do not combine large terminal and large energy increases at this stage. Lower one or both multipliers.",
                )
            )

        if approach_big_drop and terminal_big and weak_success:
            risks.append(
                cls._risk(
                    "guidance_removed_before_terminal_success",
                    "medium",
                    "Approach guidance is reduced heavily while terminal success is not yet stable.",
                    {
                        "approach_factor": approach_drop,
                        "terminal_factor": terminal_pressure,
                        "success_episode_rate": success_rate,
                    },
                    "Use a milder approach reduction or keep progress guidance until terminal success is stable.",
                )
            )

        if stability_big_drop and terminal_big:
            risks.append(
                cls._risk(
                    "stability_removed_while_terminal_increases",
                    "medium",
                    "Stability or landing-quality guidance is reduced while terminal reward is increased.",
                    {
                        "stability_factor": stability_drop,
                        "terminal_factor": terminal_pressure,
                    },
                    "Avoid reducing stability/landing-quality while increasing terminal pressure.",
                )
            )

        if high_contact_toggle and (terminal_big or energy_big):
            risks.append(
                cls._risk(
                    "contact_toggle_amplification",
                    "high",
                    "Contact toggling is already high, and the edit increases pressure that may worsen unstable contact behavior.",
                    {
                        "contact_toggle_mean": contact_toggle,
                        "terminal_factor": terminal_pressure,
                        "energy_factor": energy_pressure,
                    },
                    "Repair plan should reduce aggressive terminal/energy changes and prefer stability or continuation.",
                )
            )

        memory_risks = cls._memory_risks(edit_plan, retrieved_lessons)
        risks.extend(memory_risks)

        severity_rank = {"low": 0, "medium": 1, "high": 2}
        max_severity = "low"
        for risk in risks:
            if severity_rank.get(risk.get("severity", "low"), 0) > severity_rank[max_severity]:
                max_severity = risk.get("severity", "low")

        block_medium = bool(config.get("block_medium", False))
        audit_pass = not any(
            r.get("severity") == "high" or (block_medium and r.get("severity") == "medium")
            for r in risks
        )

        return {
            "tool": cls.name,
            "audit_pass": bool(audit_pass),
            "max_severity": max_severity,
            "risks": risks,
            "signals": {
                "success_episode_rate": success_rate,
                "stable_landing_episode_rate": stable_rate,
                "terminal_reward_paid_episode_rate": terminal_rate,
                "contact_toggle_mean": contact_toggle,
                "hack_score": hack_score,
                "terminal_factor": terminal_pressure,
                "energy_factor": energy_pressure,
                "approach_factor": approach_drop,
                "stability_factor": stability_drop,
            },
            "repair_summary": cls._repair_summary(risks),
        }

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
            "evidence": evidence,
            "repair_instruction": repair_instruction,
        }

    @classmethod
    def _max_factor(cls, edit_plan: List[Dict[str, Any]], hints: Tuple[str, ...], positive: bool) -> float:
        value = 1.0
        for edit in edit_plan or []:
            op = edit.get("operator") or edit.get("op")
            target = str(edit.get("target", "")).lower()
            if not any(h in target for h in hints):
                continue
            if op == "increase_weight":
                value = max(value, float(edit.get("factor", edit.get("value", 1.0)) or 1.0))
            elif op == "add_component":
                name = str(edit.get("component", {}).get("name", edit.get("target", ""))).lower()
                if any(h in name for h in hints):
                    value = max(value, abs(float(edit.get("component", {}).get("weight", edit.get("weight", 1.0)) or 1.0)))
        return value

    @classmethod
    def _min_factor(cls, edit_plan: List[Dict[str, Any]], hints: Tuple[str, ...], positive: bool) -> float:
        value = 1.0
        for edit in edit_plan or []:
            op = edit.get("operator") or edit.get("op")
            target = str(edit.get("target", "")).lower()
            if not any(h in target for h in hints):
                continue
            if op == "decrease_weight":
                value = min(value, float(edit.get("factor", edit.get("value", 1.0)) or 1.0))
            elif op == "disable_component":
                value = min(value, 0.0)
        return value

    @classmethod
    def _memory_risks(cls, edit_plan: List[Dict[str, Any]], lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        risks: List[Dict[str, Any]] = []
        edit_targets = {str(e.get("target", "")).lower() for e in edit_plan or []}
        edit_ops = {str(e.get("operator", e.get("op", ""))).lower() for e in edit_plan or []}

        for lesson in lessons or []:
            lesson_type = str(lesson.get("lesson_type", "")).lower()
            if "regression" not in lesson_type and "dominance" not in lesson_type:
                continue
            prior_edits = lesson.get("edit_plan", []) or []
            prior_targets = {str(e.get("target", "")).lower() for e in prior_edits if isinstance(e, dict)}
            prior_ops = {str(e.get("operator", e.get("op", ""))).lower() for e in prior_edits if isinstance(e, dict)}
            if edit_targets & prior_targets or edit_ops & prior_ops:
                risks.append(
                    cls._risk(
                        "similar_to_retrieved_regression_lesson",
                        "medium",
                        "The proposed edit overlaps with a retrieved regression lesson.",
                        {
                            "overlap_targets": sorted(edit_targets & prior_targets),
                            "overlap_operators": sorted(edit_ops & prior_ops),
                            "lesson_type": lesson_type,
                        },
                        "Ask RepairAgent to explicitly explain why this edit will not repeat the prior regression.",
                    )
                )
        return risks

    @staticmethod
    def _repair_summary(risks: List[Dict[str, Any]]) -> str:
        if not risks:
            return "No high-level behavior risk detected."
        high = [r for r in risks if r.get("severity") == "high"]
        if high:
            return "High behavior risk detected; ask RepairAgent to reduce aggressive terminal/energy changes and preserve guidance."
        return "Medium behavior risk detected; ask RepairAgent to justify or soften the edit."
