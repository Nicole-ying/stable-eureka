from __future__ import annotations

from typing import Any, Dict

from .controller import SearchController as BaseSearchController


class SearchController(BaseSearchController):
    """Search controller that preserves accept/reject behavior and exposes search mode."""

    def decide(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        decision = super().decide(*args, **kwargs)
        parent_evidence = kwargs.get("parent_evidence") if kwargs else None
        candidate_evidence = kwargs.get("candidate_evidence") if kwargs else None
        if parent_evidence is None and args:
            parent_evidence = args[0]
        if candidate_evidence is None and len(args) > 1:
            candidate_evidence = args[1]

        parent_mode = _search_mode(parent_evidence or {})
        candidate_mode = _search_mode(candidate_evidence or {})
        decision["search_mode"] = {
            "parent_next_search_mode": parent_mode,
            "candidate_next_search_mode": candidate_mode,
            "selected_next_search_mode": candidate_mode if decision.get("accepted_as_parent") else parent_mode,
            "reason": "Search mode is produced by deterministic expert_audit_pack and follows the selected next parent.",
        }
        return decision


def _search_mode(evidence: Dict[str, Any]) -> Dict[str, Any]:
    audit = evidence.get("expert_audit_pack", {}) or {}
    return audit.get("search_mode_recommendation", {}) or {}
