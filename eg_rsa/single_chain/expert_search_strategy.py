from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .json_tools import write_json, write_text
from .search_board import build_search_board_context, resolve_candidate_dir, write_search_board


def as_json_text(data: Dict[str, Any]) -> str:
    import json
    return json.dumps(data, ensure_ascii=False, indent=2)


def run_expert_search_strategy(
    llm_client: Any,
    prompt_path: str | Path,
    run_dir: str | Path,
    parent_dir: str | Path,
    best_dir: str | Path,
    next_iteration: int,
    environment_summary: Dict[str, Any],
    target_summary: Dict[str, Any],
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Ask the expert search brain which candidate should be revised next.

    The LLM chooses candidate IDs only; this function resolves IDs to existing
    directories and falls back safely if an invalid ID is returned.
    """
    run_dir = Path(run_dir)
    parent_dir = Path(parent_dir)
    best_dir = Path(best_dir)
    strategy_dir = run_dir / "search_strategy" / f"iter_{int(next_iteration):03d}"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    board = build_search_board_context(
        run_dir=run_dir,
        parent_dir=parent_dir,
        best_dir=best_dir,
        next_iteration=next_iteration,
    )
    write_search_board(strategy_dir / "search_board_context.json", board)

    prompt = Path(prompt_path).read_text(encoding="utf-8")
    # Build replacement map for the 3 placeholders
    filled_prompt = (
        prompt
        .replace("{{search_board_context_json}}", as_json_text(board))
        .replace("{{task_model_md}}", environment_summary.get("markdown_text", ""))
        .replace("{{expert_memory_context_md}}", target_summary.get("markdown_text", ""))
    )
    strategy_md = llm_client.generate(filled_prompt)
    write_text(strategy_dir / "expert_search_strategy_raw.md", strategy_md)
    write_text(strategy_dir / "expert_search_strategy.md", strategy_md)

    # Parse the Markdown output for the key decision fields
    decision = _parse_strategy_markdown(strategy_md, board)
    write_json(strategy_dir / "expert_search_strategy_decision.json", decision)

    active_parent_id = decision.get("active_parent_candidate_id")
    selected_parent_dir = resolve_candidate_dir(run_dir, active_parent_id) or parent_dir

    verified_id = decision.get("verified_elite_candidate_id")
    provisional_id = decision.get("provisional_anchor_candidate_id")
    selected_anchor_dir = resolve_candidate_dir(run_dir, verified_id) or resolve_candidate_dir(run_dir, provisional_id) or best_dir

    decision["resolved_active_parent_dir"] = str(selected_parent_dir)
    decision["resolved_anchor_dir"] = str(selected_anchor_dir)
    decision["fallback_used"] = {
        "active_parent": str(selected_parent_dir) == str(parent_dir) and active_parent_id not in board.get("candidate_ids", []),
        "anchor": str(selected_anchor_dir) == str(best_dir) and (verified_id or provisional_id) not in board.get("candidate_ids", []),
    }
    write_json(strategy_dir / "expert_search_strategy_decision.normalized.json", decision)
    write_json(run_dir / "last_expert_search_strategy_decision.json", decision)
    return selected_parent_dir, selected_anchor_dir, decision


def _parse_strategy_markdown(md: str, board: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured decision from Markdown output (JSON code block at end)."""
    import json as _json
    candidate_ids = set(board.get("candidate_ids", []) or [])

    # Try to extract JSON code block
    match = re.search(r"```json\s*\n(.*?)```", md, re.DOTALL)
    if match:
        try:
            raw = _json.loads(match.group(1))
            if isinstance(raw, dict):
                return normalize_strategy_decision(raw, board)
        except (_json.JSONDecodeError, TypeError):
            pass

    # Fallback: parse Markdown sections
    decision: Dict[str, Any] = {}
    patterns = [
        (r"\*\*Active Parent.*?\*\*\s*[-*]\s*candidate_id:\s*(\S+)", "active_parent_candidate_id"),
        (r"\*\*Verified Elite.*?\*\*\s*[-*]\s*candidate_id:\s*(\S+)", "verified_elite_candidate_id"),
        (r"\*\*Provisional Anchor.*?\*\*\s*[-*]\s*candidate_id:\s*(\S+)", "provisional_anchor_candidate_id"),
    ]
    for pat, key in patterns:
        m = re.search(pat, md, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val.lower() != "null":
                decision[key] = val

    return normalize_strategy_decision(decision, board)


def normalize_strategy_decision(decision: Dict[str, Any], board: Dict[str, Any]) -> Dict[str, Any]:
    candidate_ids = set(board.get("candidate_ids", []) or [])
    if not isinstance(decision, dict):
        decision = {}
    out = {
        "file_type": "expert_search_strategy_decision",
        "verified_elite_candidate_id": _valid_or_none(decision.get("verified_elite_candidate_id"), candidate_ids),
        "provisional_anchor_candidate_id": _valid_or_none(decision.get("provisional_anchor_candidate_id"), candidate_ids),
        "active_parent_candidate_id": _valid_or_default(decision.get("active_parent_candidate_id"), candidate_ids, board.get("current_parent_candidate_id")),
        "active_parent_role": decision.get("active_parent_role") or "frontier_candidate",
        "frontier_updates": decision.get("frontier_updates") if isinstance(decision.get("frontier_updates"), list) else [],
        "negative_edges": decision.get("negative_edges") if isinstance(decision.get("negative_edges"), list) else [],
        "closed_branch_candidate_ids": [_id for _id in (decision.get("closed_branch_candidate_ids") or []) if _id in candidate_ids],
        "promising_family_ids": decision.get("promising_family_ids") if isinstance(decision.get("promising_family_ids"), list) else [],
        "search_state_assessment": decision.get("search_state_assessment") or "Continue reward search from the most informative available parent.",
        "strategy_rationale": decision.get("strategy_rationale") or "Strategy decision normalized by framework.",
        "self_check": decision.get("self_check") if isinstance(decision.get("self_check"), dict) else {},
        "candidate_ids_available": sorted(candidate_ids),
    }
    if not out["provisional_anchor_candidate_id"] and out["active_parent_candidate_id"] != board.get("current_parent_candidate_id"):
        out["provisional_anchor_candidate_id"] = out["active_parent_candidate_id"]
    return out


def _valid_or_none(value: Any, candidate_ids: set[str]) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if value in candidate_ids else None


def _valid_or_default(value: Any, candidate_ids: set[str], default: Any) -> str:
    if value is not None and str(value) in candidate_ids:
        return str(value)
    default = str(default or "single_chain_iter0")
    return default if default in candidate_ids else sorted(candidate_ids)[0]
