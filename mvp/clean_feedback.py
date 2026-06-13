from __future__ import annotations

import io
import math
import tokenize
from typing import Any

from .semantic_audit import audit_semantic_text_bundle


# ============================================================
# Clean feedback builder
# ============================================================
#
# 目的：
#   替代 LLM 自由反思，避免 reflection 把物理语义词、
#   benchmark 先验、绝对性能误判带回 RewardCoder prompt。
#
# 原则：
#   1. 只基于已有候选的结构化结果；
#   2. 只输出相对排名与数值诊断；
#   3. 不解释 observation 维度含义；
#   4. 不声称 near-optimal / high / ceiling；
#   5. 不使用物理语义词；
#   6. 不读取、不生成任何 private evaluator 实现细节。
# ============================================================


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _fmt_float(x: Any) -> str:
    v = _safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.6g}"


def _strip_python_comments(code: str) -> str:
    """
    Remove comments from parent code before sending it back to RewardCoder.

    这样可以避免 parent code 中的自然语言注释把物理语义带入下一代。
    """
    if not code:
        return ""

    try:
        tokens = []
        reader = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(reader):
            if tok.type == tokenize.COMMENT:
                continue
            tokens.append(tok)
        return tokenize.untokenize(tokens).strip()
    except Exception:
        return code.strip()


def prepare_parent_code_for_prompt(code: str) -> str | None:
    """
    Prepare a parent reward for reuse in a clean prompt.

    如果去掉注释后仍然包含 identity / physical semantic warning，
    则不把这个 parent code 传给下一代，避免污染。
    """
    stripped = _strip_python_comments(code)
    if not stripped:
        return None

    audit = audit_semantic_text_bundle({"parent_code": stripped})
    if int(audit.get("identity_warning_count", 0)) > 0:
        return None
    if int(audit.get("semantic_term_warning_count", 0)) > 0:
        return None

    return stripped


def build_clean_reflection(top_records: list[dict[str, Any]]) -> str:
    """
    Build deterministic clean feedback.

    输出只包含匿名、结构化、相对比较信息，不调用 LLM。
    """
    if not top_records:
        return (
            "No prior clean candidates. Use only anonymous normalized feature indices. "
            "Prefer bounded functions of feature values, feature-change norms, action cost, "
            "and terminal signal. Do not assign meanings to feature dimensions."
        )

    rows = sorted(
        top_records,
        key=lambda r: _safe_float(r.get("selection_score"), -1e18),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("Deterministic clean feedback from prior candidates.")
    lines.append("Scores are only relative to observed candidates in this run; do not infer absolute optimality.")
    lines.append("Use only anonymous normalized feature indices and schema component IDs.")
    lines.append("")

    lines.append("Observed ranking by private evaluation return:")
    for rank, r in enumerate(rows, 1):
        private_eval = _safe_float(r.get("hidden_eval_return"), 0.0)
        generated_eval = _safe_float(r.get("train_mean_return"), 0.0)
        mismatch = generated_eval - private_eval

        lines.append(
            "- "
            f"rank={rank}, "
            f"id={r.get('candidate_id')}, "
            f"status={r.get('status')}, "
            f"private_eval_return={_fmt_float(private_eval)}, "
            f"generated_return={_fmt_float(generated_eval)}, "
            f"generated_minus_private={_fmt_float(mismatch)}, "
            f"repair_attempts={int(r.get('repair_attempts', 0) or 0)}, "
            f"repair_success={bool(r.get('repair_success', False))}, "
            f"identity_warning_count={int(r.get('identity_warning_count', 0) or 0)}, "
            f"semantic_term_warning_count={int(r.get('semantic_term_warning_count', 0) or 0)}"
        )

    lines.append("")
    lines.append("Schema-preserving mutation guidance:")
    lines.append("1. Prefer candidates with valid schema, low warning counts, and better relative private evaluation.")
    lines.append("2. If generated_return and private_eval_return diverge, adjust reward scaling, clipping, or component weights.")
    lines.append("3. Use anonymous feature norms and feature-change norms instead of named dimensions.")
    lines.append("4. Keep action-cost and terminal components bounded.")
    lines.append("5. Explore small component-weight changes rather than assuming any candidate is optimal.")
    lines.append("")
    lines.append("Do not use physical or benchmark-specific names for observation dimensions.")

    text = "\n".join(lines)

    # Safety check: if our deterministic text ever triggers warnings, fall back to ultra-minimal text.
    audit = audit_semantic_text_bundle({"clean_reflection": text})
    if int(audit.get("identity_warning_count", 0)) > 0 or int(audit.get("semantic_term_warning_count", 0)) > 0:
        return (
            "Prior clean candidates exist. Use relative private evaluation ranking only. "
            "Keep schema-valid components, bounded scaling, action cost, terminal signal, "
            "and anonymous feature-change functions. Do not assign meanings to feature dimensions."
        )

    return text
