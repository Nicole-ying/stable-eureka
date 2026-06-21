from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .expert_audits import build_expert_audit_pack
from .json_tools import read_json, write_json, read_text
from .target_behavior_evaluator import build_checkpoint_stability_report, build_target_behavior_report


class EvidenceBuilder:
    """Compress raw training artifacts into a compact LLM-ready evidence file."""

    @staticmethod
    def build(run_or_iter_dir: str | Path, output_path: str | Path | None = None, top_k_components: int = 8, max_episodes: int = 5) -> Dict[str, Any]:
        base = Path(run_or_iter_dir)
        training_dir = base / "training" if (base / "training").exists() else base
        reward_dir = base / "reward" if (base / "reward").exists() else base.parent / "reward"

        final_eval = _read_json_if_exists(training_dir / "final_eval.json")
        reward_trace = _read_json_if_exists(training_dir / "reward_trace.json")
        if not reward_trace and (base / "reward_trace.json").exists():
            reward_trace = _read_json_if_exists(base / "reward_trace.json")
        component_metrics = _read_json_if_exists(training_dir / "component_metrics.json")
        trajectory_summary = _read_json_if_exists(training_dir / "trajectory_summary.json")
        evals = _read_json_if_exists(training_dir / "evals.json")
        model_selection = _read_json_if_exists(training_dir / "model_selection.json")
        checkpoint_stability = _read_json_if_exists(training_dir / "checkpoint_stability_report.json")
        if not checkpoint_stability and evals:
            checkpoint_stability = build_checkpoint_stability_report(evals)
        target_behavior_report = _read_json_if_exists(training_dir / "target_behavior_report.json")
        reward_schema = _read_json_if_exists(reward_dir / "reward_schema.json")
        reward_code = _read_text_if_exists(reward_dir / "reward_code.py")
        task_model = _read_json_if_exists(base / "agents" / "task_model.json") or _read_json_if_exists(base / "environment_understanding_summary.json")

        episodes = trajectory_summary.get("episodes") or reward_trace.get("trajectory_examples") or []
        if not target_behavior_report and final_eval:
            target_behavior_report = build_target_behavior_report(
                metrics=final_eval,
                trajectories=episodes,
                max_episode_steps=None,
                fitness_score_auxiliary=final_eval.get("fitness_score"),
            )

        component_means = (
            final_eval.get("component_means")
            or component_metrics.get("component_means")
            or reward_trace.get("component_returns")
            or {}
        )
        dominant_components = _dominant_components(component_means, top_k=top_k_components)
        action_distribution = final_eval.get("action_distribution") or reward_trace.get("behavior_metrics", {}).get("action_distribution", {})
        action_space_report = final_eval.get("action_space_report") or reward_trace.get("behavior_metrics", {}).get("action_space_report", {})
        primary_metrics = {
            "fitness_score": _num(final_eval.get("fitness_score", reward_trace.get("primary_metrics", {}).get("fitness_score"))),
            "fitness_score_std": _num(final_eval.get("fitness_score_std")),
            "generated_reward": _num(final_eval.get("reward", reward_trace.get("proxy_metrics", {}).get("generated_reward"))),
            "generated_reward_std": _num(final_eval.get("reward_std")),
            "episode_length": _num(final_eval.get("episode_length", reward_trace.get("behavior_metrics", {}).get("episode_length"))),
            "episode_length_std": _num(final_eval.get("episode_length_std")),
            "success_like_terminal_rate": _num(final_eval.get("success_like_terminal_rate", reward_trace.get("behavior_metrics", {}).get("success_like_terminal_rate"))),
            "unsafe_terminal_rate": _num(final_eval.get("unsafe_terminal_rate", reward_trace.get("behavior_metrics", {}).get("unsafe_terminal_rate"))),
            "out_of_bounds_rate": _num(final_eval.get("out_of_bounds_rate", reward_trace.get("behavior_metrics", {}).get("out_of_bounds_rate"))),
        }

        episode_summaries = [_episode_digest(ep) for ep in episodes[:max_episodes]]
        automatic_hints = _automatic_hints(primary_metrics, component_means, action_distribution, action_space_report, episode_summaries, target_behavior_report, checkpoint_stability)

        evidence = {
            "file_type": "iteration_evidence",
            "source_dir": str(base),
            "candidate_id": reward_trace.get("candidate_id", base.name),
            "primary_metrics": primary_metrics,
            "reward_design_metrics": {
                "generated_minus_fitness_gap": round(
                    _num(final_eval.get("reward", reward_trace.get("proxy_metrics", {}).get("generated_reward")))
                    - _num(final_eval.get("fitness_score", reward_trace.get("primary_metrics", {}).get("fitness_score"))),
                    4,
                ),
                "objective_bonus_mean": round(_num(component_means.get("objective_bonus")), 4),
                "success_flag_mean": round(_num(component_means.get("success_flag")), 4),
                "num_positive_components": sum(
                    1 for v in component_means.values() if isinstance(v, (int, float)) and v > 1e-6
                ),
                "num_negative_components": sum(
                    1 for v in component_means.values() if isinstance(v, (int, float)) and v < -1e-6
                ),
                "dominant_component_name": dominant_components[0]["name"] if dominant_components else None,
                "dominant_component_abs_return": dominant_components[0]["abs_return"] if dominant_components else None,
                "dominance_ratio": round(
                    dominant_components[0]["abs_return"] / max(dominant_components[1]["abs_return"], 1e-9)
                    if len(dominant_components) > 1 and dominant_components[0]["abs_return"] > 0
                    else 0.0,
                    2,
                ),
                "total_component_count": len(component_means),
            },
            "policy_peak_metrics": _policy_peak_metrics(checkpoint_stability),
            "target_behavior_report": target_behavior_report,
            "checkpoint_stability_report": checkpoint_stability,
            "model_selection": model_selection or reward_trace.get("model_selection", {}),
            "behavior_summary": {
                "action_distribution": action_distribution,
                "action_space_report": action_space_report,
                "dominant_action": _dominant_action(action_distribution),
                "success_like_terminal_rate": primary_metrics["success_like_terminal_rate"],
                "unsafe_terminal_rate": primary_metrics["unsafe_terminal_rate"],
                "out_of_bounds_rate": primary_metrics["out_of_bounds_rate"],
                "dominant_behavior": _dominant_behavior(primary_metrics, action_distribution),
            },
            "component_summary": {
                "component_means": component_means,
                "dominant_components_by_abs_return": dominant_components,
                "component_count": len(component_means),
            },
            "learning_curve_summary": _summarize_evals(evals),
            "trajectory_examples": episode_summaries,
            "reward_schema_digest": _schema_digest(reward_schema),
            "reward_code_digest": _code_digest(reward_code),
            "automatic_diagnosis_hints": automatic_hints,
        }

        expert_audit_pack = build_expert_audit_pack(
            evidence=evidence,
            reward_schema=reward_schema,
            reward_code=reward_code,
            task_model=task_model,
        )
        evidence["expert_audit_pack"] = expert_audit_pack
        evidence["automatic_diagnosis_hints"] = _dedupe(
            automatic_hints + expert_audit_pack.get("automatic_hints", [])
        )

        if output_path is not None:
            write_json(output_path, evidence)
        return evidence

    @staticmethod
    def build_diagnostics_md(run_or_iter_dir: str | Path, output_path: str | Path | None = None) -> str:
        """Build a Markdown diagnostics report for LLM consumption.

        This is the primary input for the Reward Revision Agent.
        """
        base = Path(run_or_iter_dir)
        training_dir = base / "training" if (base / "training").exists() else base
        reward_dir = base / "reward" if (base / "reward").exists() else base.parent / "reward"

        final_eval = _read_json_if_exists(training_dir / "final_eval.json")
        reward_trace = _read_json_if_exists(training_dir / "reward_trace.json")
        if not reward_trace and (base / "reward_trace.json").exists():
            reward_trace = _read_json_if_exists(base / "reward_trace.json")
        component_metrics = _read_json_if_exists(training_dir / "component_metrics.json")
        trajectory_summary = _read_json_if_exists(training_dir / "trajectory_summary.json")
        evals = _read_json_if_exists(training_dir / "evals.json")
        checkpoint_stability = _read_json_if_exists(training_dir / "checkpoint_stability_report.json")
        if not checkpoint_stability and evals:
            checkpoint_stability = build_checkpoint_stability_report(evals)
        target_behavior_report = _read_json_if_exists(training_dir / "target_behavior_report.json")

        component_means = (
            final_eval.get("component_means")
            or component_metrics.get("component_means")
            or reward_trace.get("component_returns")
            or {}
        )
        episodes = trajectory_summary.get("episodes") or reward_trace.get("trajectory_examples") or []
        if not target_behavior_report and final_eval:
            target_behavior_report = build_target_behavior_report(
                metrics=final_eval,
                trajectories=episodes,
                max_episode_steps=None,
                fitness_score_auxiliary=final_eval.get("fitness_score"),
            )

        fitness = _num(final_eval.get("fitness_score", reward_trace.get("primary_metrics", {}).get("fitness_score")))
        generated = _num(final_eval.get("reward", reward_trace.get("proxy_metrics", {}).get("generated_reward")))
        ep_len = _num(final_eval.get("episode_length", reward_trace.get("behavior_metrics", {}).get("episode_length")))
        success_rate = _num(final_eval.get("success_like_terminal_rate", reward_trace.get("behavior_metrics", {}).get("success_like_terminal_rate")))
        unsafe_rate = _num(final_eval.get("unsafe_terminal_rate", reward_trace.get("behavior_metrics", {}).get("unsafe_terminal_rate")))
        oob_rate = _num(final_eval.get("out_of_bounds_rate", reward_trace.get("behavior_metrics", {}).get("out_of_bounds_rate")))
        timeout_rate = _num(target_behavior_report.get("primary_behavior_criteria", {}).get("timeout_rate")) if target_behavior_report else 0.0
        action_dist = final_eval.get("action_distribution") or reward_trace.get("behavior_metrics", {}).get("action_distribution", {})
        action_report = final_eval.get("action_space_report") or reward_trace.get("behavior_metrics", {}).get("action_space_report", {})

        # --- Build Markdown ---
        lines = [
            "# Training Diagnostics",
            "",
            "## Component Returns (per-episode mean)",
            "",
            "| Component | Mean | % of |Abs| Total | Status |",
            "|-----------|------|------|--------|",
        ]
        total_abs = sum(abs(v) for v in component_means.values() if isinstance(v, (int, float))) or 1.0
        for name in sorted(component_means.keys(), key=lambda k: abs(component_means.get(k, 0.0)), reverse=True):
            val = _num(component_means.get(name))
            pct = abs(val) / total_abs * 100 if total_abs > 0 else 0
            if abs(val) < 1e-9:
                status = "never fires"
            elif abs(val) / total_abs > 0.5 and pct > 30:
                status = "**DOMINANT**"
            elif abs(val) / total_abs > 0.3:
                status = "major"
            else:
                status = "normal"
            lines.append(f"| {name} | {val:.2f} | {pct:.0f}% | {status} |")

        lines.extend([
            "",
            "## Terminal Outcome Distribution",
            "",
            "| Outcome | Rate |",
            "|---------|------|",
            f"| Success (landed) | {success_rate:.1%} |",
            f"| Crash | {unsafe_rate:.1%} |",
            f"| Out of bounds | {oob_rate:.1%} |",
            f"| Timeout | {timeout_rate:.1%} |",
            "",
            "## Behavior Summary",
            "",
            f"- Average episode length: {ep_len:.0f} steps",
            f"- Success rate: {success_rate:.1%}",
        ])

        dom_action = ""
        dom_prob = 0.0
        for action, prob in action_dist.items():
            if _num(prob) > dom_prob:
                dom_prob = _num(prob)
                dom_action = str(action)
        dom_action_label = "passive_no_action" if dom_action == "0" and dom_prob > 0.9 else "active"
        lines.append(f"- Dominant action: {dom_action} ({dom_prob:.1%}) — {dom_action_label}")
        unused = action_report.get("unused_action_keys", [])
        if unused:
            lines.append(f"- Unused actions: {unused}")

        # Target behavior
        if target_behavior_report:
            tb = target_behavior_report
            lines.extend([
                "",
                "## Target Behavior Check",
                "",
                f"- Target satisfied: {'Yes' if tb.get('target_success') else '**No**'}",
            ])
            for gap in tb.get("target_behavior_gaps", [])[:3]:
                lines.append(f"- Gap: {str(gap)[:200]}")
            for fix in tb.get("required_reward_fixes", [])[:3]:
                lines.append(f"- Suggested fix: {str(fix)[:200]}")

        # Checkpoint stability
        if checkpoint_stability:
            cs = checkpoint_stability
            lines.extend([
                "",
                "## Checkpoint Stability",
                "",
                f"- Final fitness: {_num(cs.get('final_fitness', fitness)):.2f}",
                f"- Best checkpoint fitness: {_num(cs.get('best_fitness', fitness)):.2f}",
                f"- Transient peak risk: {'**Yes**' if cs.get('transient_peak_risk') else 'No'}",
                f"- Final uncertainty risk: {'**Yes**' if cs.get('final_uncertainty_risk') else 'No'}",
            ])

        # Alignment
        gap = generated - fitness
        lines.extend([
            "",
            "## Reward-Behavior Alignment",
            "",
            f"- True fitness score: {fitness:.2f}",
            f"- Agent-generated reward: {generated:.2f}",
            f"- Gap (generated - fitness): {gap:.2f}",
        ])
        if gap > 100:
            lines.append("- ⚠️ Large gap: agent collects much more reward than true fitness suggests.")
            lines.append("  Possible proxy misalignment or reward farming.")

        # Trajectory examples
        lines.extend([
            "",
            "## Trajectory Examples",
        ])
        for ep in episodes[:3]:
            comp = ep.get("component_returns", {})
            top_comp = sorted([(k, v) for k, v in comp.items() if isinstance(v, (int, float))], key=lambda kv: abs(kv[1]), reverse=True)[:3]
            top_str = ", ".join([f"{k}={v:.1f}" for k, v in top_comp])
            fs = ep.get("final_state") or []
            fs_str = f"x={_num(fs[0]):.2f} y={_num(fs[1]):.2f} vy={_num(fs[3]):.2f}" if len(fs) >= 4 else "N/A"
            lines.append(f"- Ep {ep.get('episode_id','?')}: len={ep.get('length','?')}, "
                         f"return={_num(ep.get('return_generated')):.1f}, "
                         f"terminal={ep.get('terminal_classification','?')}, "
                         f"final: {fs_str}, "
                         f"top: {top_str}")

        markdown = "\n".join(lines)
        if output_path is not None:
            Path(output_path).write_text(markdown, encoding="utf-8")
        return markdown


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(path)}


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return read_text(path)
    except Exception:
        return ""


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _dominant_components(component_means: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    items: List[Tuple[str, float]] = []
    for key, value in component_means.items():
        try:
            if key in {"fitness_score", "total_reward"}:
                continue
            items.append((key, float(value)))
        except Exception:
            continue
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"name": key, "mean_return": value, "abs_return": abs(value)} for key, value in items[:top_k]]


def _dominant_action(action_distribution: Dict[str, Any]) -> Dict[str, Any]:
    if not action_distribution:
        return {"action": None, "probability": 0.0}
    action, prob = max(action_distribution.items(), key=lambda kv: _num(kv[1]))
    return {"action": action, "probability": _num(prob)}


def _dominant_behavior(metrics: Dict[str, float], action_distribution: Dict[str, Any]) -> str:
    dom = _dominant_action(action_distribution)
    if dom["probability"] >= 0.95 and str(dom["action"]) in {"0", "0.0"}:
        return "passive_no_action_policy"
    if metrics.get("success_like_terminal_rate", 0.0) <= 0.01 and metrics.get("episode_length", 0.0) < 120:
        return "early_failure_or_uncontrolled_terminal"
    if metrics.get("success_like_terminal_rate", 0.0) <= 0.01 and metrics.get("episode_length", 0.0) > 800:
        return "hovering_or_timeout_policy"
    if metrics.get("success_like_terminal_rate", 0.0) > 0.5:
        return "promising_success_like_policy"
    return "unclear_policy"


def _episode_digest(ep: Dict[str, Any]) -> Dict[str, Any]:
    final_state = ep.get("final_state") or []
    return {
        "episode_id": ep.get("episode_id"),
        "length": ep.get("length"),
        "return_generated": ep.get("return_generated"),
        "return_fitness": ep.get("return_fitness"),
        "terminal_classification": ep.get("terminal_classification"),
        "final_state": final_state,
        "action_histogram": ep.get("action_histogram", {}),
        "component_returns_top_abs": _dominant_components(ep.get("component_returns", {}), top_k=6),
    }


def _summarize_evals(evals: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in ["timesteps", "reward", "reward_std", "fitness_score", "fitness_score_std", "episode_length", "episode_length_std", "success_like_terminal_rate", "unsafe_terminal_rate", "out_of_bounds_rate"]:
        values = evals.get(key)
        if isinstance(values, list) and values:
            summary[key] = {
                "first": values[0],
                "last": values[-1],
                "best": max(values) if all(isinstance(v, (int, float)) for v in values) else None,
                "num_points": len(values),
            }
    if isinstance(evals.get("action_space_report"), list) and evals.get("action_space_report"):
        summary["action_space_report_last"] = evals["action_space_report"][-1]
    return summary


def _schema_digest(schema: Any) -> Dict[str, Any]:
    if not schema:
        return {}
    if isinstance(schema, list):
        return {
            "component_ids": [c.get("id") for c in schema if isinstance(c, dict)],
            "active_reward_terms": [c.get("id") for c in schema if isinstance(c, dict) and c.get("type") in ("active", "active_objective", "objective")],
            "diagnostic_terms": [c.get("id") for c in schema if isinstance(c, dict) and c.get("type") in ("diagnostic", "regularizer")],
            "dormant_terms": [c.get("id") for c in schema if isinstance(c, dict) and c.get("type") == "dormant"],
            "design_principle": [c.get("description") for c in schema if isinstance(c, dict) and c.get("description")][:3],
        }
    return {
        "version": schema.get("version") or schema.get("schema_version"),
        "design_principle": schema.get("design_principle"),
        "active_reward_terms": schema.get("active_reward_terms"),
        "diagnostic_terms": schema.get("diagnostic_terms"),
        "dormant_terms": schema.get("dormant_terms"),
        "component_ids": [c.get("id") for c in schema.get("component_catalog", schema.get("components", [])) if isinstance(c, dict)],
    }


def _code_digest(code: str) -> Dict[str, Any]:
    if not code:
        return {}
    lines = [line.rstrip() for line in code.splitlines()]
    return {
        "num_lines": len(lines),
        "contains_internal_state": "hasattr(self" in code or "self._" in code,
        "individual_reward_keys_hint": _extract_individual_reward_keys(code),
    }


def _extract_individual_reward_keys(code: str) -> List[str]:
    keys: List[str] = []
    marker = "individual_reward["
    for part in code.split(marker)[1:]:
        quote = "'" if "'" in part[:3] else '"'
        try:
            key = part.split(quote, 2)[1]
            if key not in keys:
                keys.append(key)
        except Exception:
            continue
    return keys


def _automatic_hints(
    metrics: Dict[str, float],
    component_means: Dict[str, Any],
    action_distribution: Dict[str, Any],
    action_space_report: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    target_behavior_report: Dict[str, Any],
    checkpoint_stability: Dict[str, Any],
) -> List[str]:
    hints: List[str] = []
    dom = _dominant_action(action_distribution)
    if dom["probability"] >= 0.95:
        hints.append(f"Policy collapsed to a single dominant action: {dom['action']} with probability {dom['probability']:.3f}.")
    if str(dom["action"]) in {"0", "0.0"} and dom["probability"] >= 0.95:
        hints.append("No-action/passive policy likely: reward may over-penalize control or fail to provide reachable progress signals.")
    unused = action_space_report.get("unused_action_keys") or []
    if unused:
        hints.append("Unused discrete actions detected during evaluation: " + ", ".join(map(str, unused)) + ". Check whether the reward makes necessary actions unattractive.")
    if metrics.get("success_like_terminal_rate", 0.0) <= 0.01:
        hints.append("Final policy did not discover success-like terminal behavior. Treat fitness as auxiliary, not proof of reward quality.")
    if _num(component_means.get("fuel_cost")) == 0.0 and str(dom["action"]) in {"0", "0.0"}:
        hints.append("Fuel cost is zero because the policy never uses engines/actions that consume fuel.")
    dominant = _dominant_components(component_means, top_k=3)
    if dominant:
        hints.append("Dominant reward components by absolute return: " + ", ".join(f"{d['name']}={d['mean_return']:.3f}" for d in dominant))
    if any((ep.get("length") or 0) < 120 for ep in episodes) and metrics.get("success_like_terminal_rate", 0.0) <= 0.01:
        hints.append("Episodes terminate early without success-like behavior; consider exploration-friendly progress signals before adding stronger penalties.")
    if target_behavior_report and not target_behavior_report.get("target_success", False):
        gaps = target_behavior_report.get("target_behavior_gaps", []) or []
        hints.append("Target behavior gap: " + "; ".join(map(str, gaps[:3])))
    if checkpoint_stability.get("transient_peak_risk"):
        hints.append("Checkpoint stability risk: training showed a transient peak but final behavior regressed; do not use peak fitness as reward-quality evidence.")
    return hints


def _policy_peak_metrics(checkpoint_stability: Dict[str, Any]) -> Dict[str, Any]:
    if not checkpoint_stability or not checkpoint_stability.get("available"):
        return {}
    return {
        "diagnostic_only": True,
        "best_fitness": checkpoint_stability.get("best_fitness"),
        "best_success_like_terminal_rate": checkpoint_stability.get("best_success_like_terminal_rate"),
        "best_timestep": checkpoint_stability.get("best_timestep"),
        "best_final_fitness_gap": checkpoint_stability.get("best_final_fitness_gap"),
        "success_rate_drop": checkpoint_stability.get("success_rate_drop"),
        "transient_peak_risk": checkpoint_stability.get("transient_peak_risk"),
    }


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
