from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .json_tools import write_json


def build_expert_memory_context(
    current_evidence: Dict[str, Any],
    best_evidence: Dict[str, Any] | None = None,
    retrieved_memory: List[Dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    behavior = current_evidence.get("behavior_summary", {}) or {}
    metrics = current_evidence.get("primary_metrics", {}) or {}
    comp = current_evidence.get("component_summary", {}) or {}
    action = behavior.get("dominant_action", {}) or {}
    action_space_report = behavior.get("action_space_report", {}) or {}
    action_id = str(action.get("action"))
    action_prob = _num(action.get("probability"))
    success_rate = _num(behavior.get("success_like_terminal_rate", metrics.get("success_like_terminal_rate")))
    episode_len = _num(metrics.get("episode_length"))
    components = comp.get("component_means", {}) or {}

    hard_constraints: List[str] = []
    diagnosis_questions: List[str] = []
    failed_patterns: List[Dict[str, Any]] = []

    mostly_negative = _mostly_negative_components(components)
    if action_id in {"0", "0.0", "None"} and action_prob >= 0.90 and success_rate <= 0.01:
        failed_patterns.append({
            "pattern_id": "passive_dominant_action",
            "evidence": {"dominant_action": action_id, "probability": action_prob, "success_rate": success_rate},
            "interpretation": "The current reward makes minimal action competitive with trying the task.",
        })
        hard_constraints.extend([
            "Do not strengthen action, fuel, or time costs under the same dominant-action behavior.",
            "Add or strengthen a reachable positive progress signal tied to the task objective.",
            "If task achievement is detectable, keep it active rather than diagnostic-only.",
            "Explain why the next reward should change the action distribution away from the current dominant action."
        ])

    unused_actions = action_space_report.get("unused_action_keys") or []
    if unused_actions:
        failed_patterns.append({
            "pattern_id": "unused_discrete_actions",
            "evidence": {
                "unused_action_keys": unused_actions,
                "action_space_report": action_space_report,
                "success_rate": success_rate,
            },
            "interpretation": "Some available actions were never selected during evaluation; a necessary control mode may be unattractive under the reward.",
        })
        hard_constraints.extend([
            "Explicitly analyze every unused discrete action and whether any is necessary for task success.",
            "Do not only reduce penalties; make the reward create a reason to try currently unused necessary actions.",
            "The next expected_behavior must predict which unused action usage should increase and why."
        ])

    if episode_len > 0 and episode_len < 150 and success_rate <= 0.01 and mostly_negative:
        failed_patterns.append({
            "pattern_id": "short_episode_dense_penalty",
            "evidence": {"episode_length": episode_len, "success_rate": success_rate},
            "interpretation": "Longer attempts may accumulate more penalty than early termination.",
        })
        hard_constraints.extend([
            "Do not repeat a reward made mainly of absolute dense penalties.",
            "Prefer progress-delta or milestone-style feedback over only negative state costs.",
            "Make useful longer attempts able to score better than short bad episodes."
        ])

    dominant = comp.get("dominant_components_by_abs_return", []) or []
    if dominant:
        top = dominant[0]
        top_abs = _num(top.get("abs_return"))
        second_abs = _num(dominant[1].get("abs_return")) if len(dominant) > 1 else 0.0
        if top_abs > 0 and (second_abs == 0.0 or top_abs >= 3.0 * max(second_abs, 1e-9)):
            failed_patterns.append({
                "pattern_id": "single_component_dominance",
                "evidence": {"top_component": top},
                "interpretation": "One component may dominate the training signal.",
            })
            hard_constraints.append("Estimate component episode-scale budgets and reduce any auxiliary term that dominates the objective signal.")

    diagnosis_questions.extend([
        "What behavior is the policy optimizing under the current reward?",
        "What behavior required for the task is being avoided?",
        "Which component makes the avoided behavior costly?",
        "Which positive signal is missing, too sparse, or too weak?",
        "Are any available actions unused, and could an unused action be required for success?",
        "Which previous edit pattern must not be repeated?"
    ])

    retrieved_memory = retrieved_memory or []
    for item in retrieved_memory:
        acc = item.get("acceptance", {}) or {}
        if acc.get("rejected"):
            lesson = item.get("lesson")
            if lesson:
                hard_constraints.append("Avoid repeating rejected edit pattern: " + str(lesson)[:400])

    context = {
        "file_type": "expert_memory_context",
        "diagnosis_questions": diagnosis_questions,
        "failed_patterns": failed_patterns,
        "hard_constraints_for_next_revision": _dedupe(hard_constraints),
        "component_balance_report": _component_balance_report(components, dominant),
        "action_space_report": action_space_report,
        "best_reference": _best_reference(best_evidence),
    }
    if output_path is not None:
        write_json(output_path, context)
    return context


def build_memory_context_md(
    current_evidence: Dict[str, Any],
    retrieved_memory: List[Dict[str, Any]] | None = None,
    strategy_decision: Dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Build a structured Markdown memory context for the Revision Agent.

    Designed for LLM readability: short sections, bullet lists, clear ❌/✅ markers.
    """
    retrieved_memory = retrieved_memory or []
    strategy = strategy_decision or {}

    lines = [
        "# Memory Context — Read Before Editing",
        "",
        "Below are patterns learned from past iterations. Violating these will waste training budget.",
        "",
    ]

    # -- Revision Brief from Search Strategy (highest priority) --
    revision_brief = strategy.get("revision_brief", {}) or {}
    if revision_brief:
        lines.append("## 🎯 Revision Brief (from Search Strategist)")
        lines.append("")
        req_dir = revision_brief.get("required_direction", "")
        if req_dir:
            lines.append(f"**Required direction**: {req_dir}")
            lines.append("")
        forbidden = revision_brief.get("forbidden_edit_types", []) or []
        if forbidden:
            lines.append("**FORBIDDEN edit types**:")
            for fb in forbidden:
                lines.append(f"- ❌ {fb}")
            lines.append("")
        succ = revision_brief.get("success_criteria", "")
        if succ:
            lines.append(f"**Success criteria**: {succ}")
            lines.append("")

    # -- Strategy state --
    if strategy:
        lines.append("## Search State")
        search_assessment = strategy.get("search_state_assessment", "")
        if search_assessment:
            lines.append(f"{search_assessment[:600]}")
            lines.append("")
        negative = strategy.get("negative_edges", []) or []
        if negative:
            lines.append("### Known Bad Edit Patterns")
            lines.append("")
            lines.append("| From | To | Severity | Pattern |")
            lines.append("|------|----|----------|---------|")
            for ne in negative[:8]:
                tier = ne.get("tier", "SOFT")
                pattern = ne.get("avoid_pattern", "")[:150]
                lines.append(f"| {ne.get('from','?')} | {ne.get('to','?')} | {tier} | {pattern} |")
            lines.append("")

    # -- Behavioral diagnosis --
    behavior = current_evidence.get("behavior_summary", {}) or {}
    metrics = current_evidence.get("primary_metrics", {}) or {}
    success_rate = _num(behavior.get("success_like_terminal_rate", metrics.get("success_like_terminal_rate")))
    episode_len = _num(metrics.get("episode_length"))
    oob_rate = _num(metrics.get("out_of_bounds_rate", behavior.get("out_of_bounds_rate")))
    unsafe_rate = _num(metrics.get("unsafe_terminal_rate", behavior.get("unsafe_terminal_rate")))

    lines.append("## Current Policy Behavior")
    lines.append("")
    lines.append(f"- Success rate: {success_rate:.1%}")
    lines.append(f"- Episode length: {episode_len:.0f} steps")
    lines.append(f"- Out-of-bounds rate: {oob_rate:.1%}")
    lines.append(f"- Unsafe terminal rate: {unsafe_rate:.1%}")
    dominant_behavior = behavior.get("dominant_behavior", "unknown")
    lines.append(f"- Dominant behavior: {dominant_behavior}")
    lines.append("")

    action_id = str((behavior.get("dominant_action", {}) or {}).get("action", ""))
    action_prob = _num((behavior.get("dominant_action", {}) or {}).get("probability", 0))
    if action_id in ("0", "0.0") and action_prob >= 0.90:
        lines.append("⚠️ **Action collapse**: policy uses action 0 >90% of the time. Penalty-dominated reward.")
        lines.append("")

    unused = (behavior.get("action_space_report", {}) or {}).get("unused_action_keys", [])
    if unused:
        lines.append(f"⚠️ **Unused actions**: {unused} — these are NEVER selected. If any is required for success, the reward must incentivize it.")
        lines.append("")

    # -- Component balance --
    comp = current_evidence.get("component_summary", {}) or {}
    comp_means = comp.get("component_means", {}) or {}
    dominant = comp.get("dominant_components_by_abs_return", []) or []
    if dominant:
        lines.append("## Reward Component Balance")
        lines.append("")
        lines.append("| Component | Abs Return |")
        lines.append("|-----------|-----------|")
        for d in dominant[:5]:
            name = d.get("name", "?")
            val = d.get("abs_return", 0)
            lines.append(f"| {name} | {val:.1f} |")
        lines.append("")
        if len(dominant) >= 2:
            top_val = _num(dominant[0].get("abs_return"))
            second_val = _num(dominant[1].get("abs_return"))
            if top_val > 3.0 * max(second_val, 1e-9):
                lines.append(f"⚠️ **{dominant[0].get('name', '?')}** dominates the reward budget ({top_val:.0f} vs {second_val:.0f}). The policy is optimizing this component, not the task.")
                lines.append("")

    # -- Past iteration lessons --
    negative_lessons = []
    positive_lessons = []
    for item in retrieved_memory:
        reuse = item.get("reuse_policy", "")
        lesson = item.get("lesson", "")
        if not lesson:
            continue
        edit = item.get("edit_summary", {}) or {}
        what = edit.get("what_changed", "") or ""
        if reuse in ("negative_constraint",):
            negative_lessons.append((lesson, what))
        elif reuse in ("provisional_anchor", "positive_reference"):
            positive_lessons.append((lesson, what))

    if negative_lessons:
        lines.append("## ❌ Failed Edits — DO NOT REPEAT")
        lines.append("")
        for i, (lesson, what) in enumerate(negative_lessons[:5]):
            # Extract the core lesson more concisely
            short = _summarize_memory_lesson(lesson, what)
            lines.append(f"{i+1}. {short}")
        lines.append("")

    if positive_lessons:
        lines.append("## ✅ Successful Directions — PRESERVE")
        lines.append("")
        for i, (lesson, what) in enumerate(positive_lessons[:3]):
            short = _summarize_memory_lesson(lesson, what)
            lines.append(f"{i+1}. {short}")
        lines.append("")

    markdown = "\n".join(lines)
    if output_path is not None:
        Path(output_path).write_text(markdown, encoding="utf-8")
    return markdown


def _summarize_memory_lesson(lesson: str, what_changed: str) -> str:
    """Extract a concise summary from a verbose memory lesson string."""
    # If we have a structured what_changed, use it directly
    if what_changed and len(what_changed) > 10:
        return what_changed[:300]
    # Otherwise extract key parts from the lesson
    # Remove redundant prefixes
    for prefix in ["A ", "An "]:
        if lesson.startswith(prefix):
            # Try to extract: "A X reward revision from behavior A to B VERDICT. ..."
            parts = lesson.split(". ")
            if len(parts) >= 2:
                return ". ".join(parts[1:3])[:300] if len(parts) >= 3 else parts[1][:300]
    return lesson[:300]


def _component_balance_report(components: Dict[str, Any], dominant: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = []
    for k, v in components.items():
        try:
            vals.append((k, float(v)))
        except Exception:
            continue
    pos = [x for x in vals if x[1] > 0]
    neg = [x for x in vals if x[1] < 0]
    return {
        "num_positive_components": len(pos),
        "num_negative_components": len(neg),
        "mostly_negative": len(neg) > 0 and len(pos) == 0,
        "dominant_components_by_abs_return": dominant[:5],
    }


def _best_reference(best: Dict[str, Any] | None) -> Dict[str, Any]:
    if not best:
        return {}
    return {
        "candidate_id": best.get("candidate_id"),
        "primary_metrics": best.get("primary_metrics", {}),
        "behavior_summary": best.get("behavior_summary", {}),
    }


def _mostly_negative_components(components: Dict[str, Any]) -> bool:
    vals = []
    for v in components.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return False
    positives = [v for v in vals if v > 1e-9]
    negatives = [v for v in vals if v < -1e-9]
    return len(negatives) > 0 and len(positives) == 0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _dedupe(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
