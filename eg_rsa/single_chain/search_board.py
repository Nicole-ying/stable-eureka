from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .json_tools import read_json, read_text, write_json


def candidate_id_from_dir(run_dir: Path, candidate_dir: Path) -> str:
    candidate_dir = Path(candidate_dir)
    try:
        if candidate_dir.resolve() == run_dir.resolve():
            return "single_chain_iter0"
    except Exception:
        if candidate_dir == run_dir:
            return "single_chain_iter0"
    match = re.fullmatch(r"iter_(\d+)", candidate_dir.name)
    if match:
        return f"iter_{int(match.group(1)):03d}"
    evidence = _read_json_if_exists(candidate_dir / "iteration_evidence.json")
    return str(evidence.get("candidate_id") or candidate_dir.name)


def resolve_candidate_dir(run_dir: Path, candidate_id: str | None) -> Optional[Path]:
    if not candidate_id:
        return None
    candidate_id = str(candidate_id).strip()
    if candidate_id in {"single_chain_iter0", "iter_000", run_dir.name, "bootstrap"}:
        return run_dir
    match = re.fullmatch(r"iter_(\d+)", candidate_id)
    if match:
        path = run_dir / "iterations" / f"iter_{int(match.group(1)):03d}"
        if path.exists():
            return path
    return None


def list_candidate_dirs(run_dir: Path) -> List[Path]:
    out = [run_dir]
    iterations_dir = run_dir / "iterations"
    if iterations_dir.exists():
        for child in sorted(iterations_dir.iterdir(), key=lambda p: p.name):
            if child.is_dir() and re.fullmatch(r"iter_\d+", child.name):
                out.append(child)
    return out


def build_candidate_dossier(run_dir: Path, candidate_dir: Path, include_reward_code: bool = True) -> Dict[str, Any]:
    candidate_id = candidate_id_from_dir(run_dir, candidate_dir)
    evidence = _read_json_if_exists(candidate_dir / "iteration_evidence.json")
    if not evidence and candidate_dir == run_dir:
        evidence = _read_json_if_exists(candidate_dir / "iteration_evidence.json")
    controller_decision = _read_json_if_exists(candidate_dir / "controller_decision.json")
    reward_guard = _read_json_if_exists(candidate_dir / "reward_guard_summary.json")
    semantic_noop = _read_json_if_exists(candidate_dir / "reward" / "semantic_noop_report.json")
    revision_metadata = _read_json_if_exists(candidate_dir / "revision_metadata.json")
    reward_schema = _read_json_if_exists(candidate_dir / "reward" / "reward_schema.json")
    reward_code = _read_text_if_exists(candidate_dir / "reward" / "reward_code.py") if include_reward_code else ""

    primary = evidence.get("primary_metrics", {}) or _root_primary_metrics(candidate_dir)
    behavior = evidence.get("behavior_summary", {}) or _root_behavior_metrics(candidate_dir)
    target_report = evidence.get("target_behavior_report", {}) or _read_json_if_exists(candidate_dir / "training" / "target_behavior_report.json")
    stability = evidence.get("checkpoint_stability_report", {}) or _read_json_if_exists(candidate_dir / "training" / "checkpoint_stability_report.json")

    return {
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "role_hints": _role_hints(evidence, controller_decision, primary, behavior, target_report, stability),
        "primary_metrics": primary,
        "behavior_summary": behavior,
        "target_behavior_report": _compact_target_report(target_report),
        "checkpoint_stability_report": _compact_stability_report(stability),
        "controller_decision": _compact_controller_decision(controller_decision),
        "revision_metadata": revision_metadata,
        "reward_guard_summary": _compact_reward_guard(reward_guard),
        "semantic_noop_report": semantic_noop,
        "reward_schema": reward_schema,
        "reward_code": reward_code,
    }


def build_search_board_context(
    run_dir: str | Path,
    parent_dir: str | Path,
    best_dir: str | Path,
    next_iteration: int,
    max_full_code_candidates: int = 8,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    parent_dir = Path(parent_dir)
    best_dir = Path(best_dir)
    candidate_dirs = list_candidate_dirs(run_dir)
    selected_for_full_code = _select_full_code_candidates(run_dir, candidate_dirs, parent_dir, best_dir, max_full_code_candidates)
    dossiers = []
    for candidate_dir in candidate_dirs:
        dossiers.append(
            build_candidate_dossier(
                run_dir,
                candidate_dir,
                include_reward_code=candidate_dir in selected_for_full_code,
            )
        )
    board = {
        "file_type": "expert_search_board_context",
        "next_iteration": next_iteration,
        "current_parent_candidate_id": candidate_id_from_dir(run_dir, parent_dir),
        "current_best_candidate_id": candidate_id_from_dir(run_dir, best_dir),
        "candidate_count": len(dossiers),
        "candidate_ids": [d["candidate_id"] for d in dossiers],
        "candidate_dossiers": dossiers,
        "decision_contract": {
            "hard_rules": [
                "Choose only candidate_ids present in candidate_ids.",
                "Do not select a semantic no-op or invalid reward as active parent.",
                "Do not call generated_reward the framework objective; it is diagnostic only.",
                "Do not treat peak checkpoint metrics as verified reward quality.",
            ],
            "expert_responsibilities": [
                "Decide verified_elite_candidate_id if any reward is truly stable and target-aligned.",
                "Decide provisional_anchor_candidate_id for the best promising but not fully verified direction.",
                "Decide active_parent_candidate_id whose full reward code should be revised next.",
                "Mark closed_branch_candidate_ids and negative_edges to avoid repeated bad edits.",
                "Assess the current search state — what problem does the best reward have?",
            ],
        },
    }
    return board


def write_search_board(path: str | Path, board: Dict[str, Any]) -> None:
    write_json(path, board)


def _select_full_code_candidates(run_dir: Path, candidate_dirs: List[Path], parent_dir: Path, best_dir: Path, limit: int) -> set[Path]:
    scored: List[tuple[float, Path]] = []
    for path in candidate_dirs:
        evidence = _read_json_if_exists(path / "iteration_evidence.json")
        primary = evidence.get("primary_metrics", {}) or _root_primary_metrics(path)
        behavior = evidence.get("behavior_summary", {}) or _root_behavior_metrics(path)
        score = _num(primary.get("fitness_score")) + 200.0 * _num(behavior.get("success_like_terminal_rate"))
        if path in {parent_dir, best_dir, run_dir}:
            score += 10000.0
        scored.append((score, path))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return {path for _, path in scored[: max(1, int(limit))]}


def _role_hints(evidence: Dict[str, Any], controller: Dict[str, Any], primary: Dict[str, Any], behavior: Dict[str, Any], target: Dict[str, Any], stability: Dict[str, Any]) -> Dict[str, Any]:
    success = _num(behavior.get("success_like_terminal_rate", primary.get("success_like_terminal_rate")))
    timeout = _num((target.get("primary_behavior_criteria", {}) or {}).get("timeout_rate"))
    fitness = _num(primary.get("fitness_score"))
    accepted_parent = bool(controller.get("accepted_as_parent"))
    accepted_elite = bool(controller.get("accepted_as_elite"))
    rejected = bool(controller.get("rejected"))
    return {
        "accepted_as_parent_by_controller": accepted_parent,
        "accepted_as_elite_by_controller": accepted_elite,
        "rejected_by_controller": rejected,
        "has_partial_success_signal": success > 0.0,
        "timeout_dominated": bool(timeout >= 0.8 and success <= 0.01),
        "transient_peak_risk": bool(stability.get("transient_peak_risk")),
        "target_success": bool(target.get("target_success")),
        "fitness_score": fitness,
    }


def _compact_target_report(report: Dict[str, Any]) -> Dict[str, Any]:
    if not report:
        return {}
    return {
        "target_success": report.get("target_success"),
        "quality_label": report.get("quality_label"),
        "evidence_confidence": report.get("evidence_confidence"),
        "primary_behavior_criteria": report.get("primary_behavior_criteria"),
        "auxiliary_scores": report.get("auxiliary_scores"),
        "auxiliary_score_stability": report.get("auxiliary_score_stability"),
        "behavior_diagnostics": report.get("behavior_diagnostics"),
        "target_behavior_gaps": report.get("target_behavior_gaps"),
        "required_reward_fixes": report.get("required_reward_fixes"),
        "target_behavior_contract": report.get("target_behavior_contract"),
    }


def _compact_stability_report(report: Dict[str, Any]) -> Dict[str, Any]:
    if not report:
        return {}
    keys = [
        "available",
        "best_checkpoint_is_diagnostic_only",
        "best_fitness",
        "final_fitness",
        "final_fitness_std",
        "best_final_fitness_gap",
        "best_success_like_terminal_rate",
        "final_success_like_terminal_rate",
        "success_rate_drop",
        "last_k_eval",
        "transient_peak_risk",
        "reward_behavior_divergence",
        "final_uncertainty_risk",
        "unstable_last_k_risk",
        "hard_constraints",
    ]
    return {k: report.get(k) for k in keys if k in report}


def _compact_controller_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    if not decision:
        return {}
    return {
        "parent_candidate_id": decision.get("parent_candidate_id"),
        "candidate_id": decision.get("candidate_id"),
        "elite_candidate_id": decision.get("elite_candidate_id"),
        "scores": decision.get("scores"),
        "accepted_as_parent": decision.get("accepted_as_parent"),
        "accepted_as_elite": decision.get("accepted_as_elite"),
        "elite_status": decision.get("elite_status"),
        "rejected": decision.get("rejected"),
        "rejection_reasons": decision.get("rejection_reasons"),
        "next_search_parent": decision.get("next_search_parent"),
        "expert_acceptance_gate": {
            "reward_elite_allowed": (decision.get("expert_acceptance_gate", {}) or {}).get("reward_elite_allowed"),
            "elite_status": (decision.get("expert_acceptance_gate", {}) or {}).get("elite_status"),
            "parent_allowed": (decision.get("expert_acceptance_gate", {}) or {}).get("parent_allowed"),
            "policy_reference_allowed": (decision.get("expert_acceptance_gate", {}) or {}).get("policy_reference_allowed"),
            "reasons": (decision.get("expert_acceptance_gate", {}) or {}).get("reasons"),
            "advisory_reasons": (decision.get("expert_acceptance_gate", {}) or {}).get("advisory_reasons"),
            "risk_metrics": (decision.get("expert_acceptance_gate", {}) or {}).get("risk_metrics"),
        },
    }


def _compact_reward_guard(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {}
    attempts = summary.get("attempts") or []
    last = attempts[-1] if attempts else summary.get("last_reports", {}) or {}
    return {
        "valid": summary.get("valid"),
        "repair_attempts_used": summary.get("repair_attempts_used"),
        "last_attempt": {
            "validation_valid": last.get("validation_valid"),
            "runtime_safety_valid": last.get("runtime_safety_valid"),
            "static_validation_valid": last.get("static_validation_valid"),
            "semantic_noop_valid": last.get("semantic_noop_valid"),
            "semantic_noop_edit": last.get("semantic_noop_edit"),
            "smoke_test_valid": last.get("smoke_test_valid"),
        },
    }


def _root_primary_metrics(path: Path) -> Dict[str, Any]:
    trace = _read_json_if_exists(path / "reward_trace.json")
    primary = trace.get("primary_metrics", {}) or {}
    proxy = trace.get("proxy_metrics", {}) or {}
    behavior = trace.get("behavior_metrics", {}) or {}
    return {
        "fitness_score": primary.get("fitness_score"),
        "generated_reward": proxy.get("generated_reward"),
        "episode_length": behavior.get("episode_length"),
        "success_like_terminal_rate": behavior.get("success_like_terminal_rate"),
        "unsafe_terminal_rate": behavior.get("unsafe_terminal_rate"),
        "out_of_bounds_rate": behavior.get("out_of_bounds_rate"),
    }


def _root_behavior_metrics(path: Path) -> Dict[str, Any]:
    trace = _read_json_if_exists(path / "reward_trace.json")
    return trace.get("behavior_metrics", {}) or {}


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
