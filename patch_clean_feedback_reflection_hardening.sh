#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_clean_feedback_reflection_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/agents.py \
  mvp/clean_feedback.py \
  scripts/run_clean_hardened_single_seed.sh
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/6] write mvp/clean_feedback.py..."
cat > mvp/clean_feedback.py <<'PY'
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
PY

echo "[3/6] patch mvp/agents.py..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/agents.py")
s = p.read_text(encoding="utf-8")

if "from .clean_feedback import build_clean_reflection, prepare_parent_code_for_prompt" not in s:
    s = s.replace(
        "from .models import ModelGateway\n",
        "from .models import ModelGateway\n"
        "from .clean_feedback import build_clean_reflection, prepare_parent_code_for_prompt\n",
    )

old_parent = '''        parent_block = "\\n\\n".join(
            [f"Parent {i + 1}:\\n```python\\n{c}\\n```" for i, c in enumerate(parent_codes)]
        ) or "No clean parent code yet."
'''

new_parent = '''        safe_parent_codes = []
        for c in parent_codes:
            safe_code = prepare_parent_code_for_prompt(c)
            if safe_code:
                safe_parent_codes.append(safe_code)

        parent_block = "\\n\\n".join(
            [f"Parent {i + 1}:\\n```python\\n{c}\\n```" for i, c in enumerate(safe_parent_codes)]
        ) or "No anonymous parent code available."
'''

if old_parent not in s:
    if "safe_parent_codes = []" in s:
        print("RewardCoderAgent parent sanitization already present; skip.")
    else:
        raise SystemExit("ERROR: parent_block section not found in mvp/agents.py")
else:
    s = s.replace(old_parent, new_parent)

# Replace ReflectionAgent with deterministic clean feedback implementation.
marker = "class ReflectionAgent:"
if marker not in s:
    raise SystemExit("ERROR: ReflectionAgent class not found in mvp/agents.py")

prefix = s[:s.index(marker)]
new_reflection = r'''class ReflectionAgent:
    """
    Deterministic clean feedback agent.

    不再调用 LLM 自由生成 reflection，避免把物理语义词、
    benchmark 先验、绝对性能判断带回下一代 RewardCoder prompt。
    """

    def __init__(self, model: ModelGateway):
        self.model = model

    def summarize(self, top_records: list[dict]) -> str:
        return build_clean_reflection(top_records)
'''

s = prefix + new_reflection + "\n"
p.write_text(s, encoding="utf-8")
PY

echo "[4/6] write single-seed hardened smoke script..."
cat > scripts/run_clean_hardened_single_seed.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set."
  echo "Please run:"
  echo "  export DEEPSEEK_API_KEY='your_key_here'"
  exit 1
fi

MODE="${1:-cartpole}"

run_cartpole() {
  WORKSPACE="runs/clean_cartpole_deepseek_seed0_g2p2_t8k"
  CONFIG="mvp/configs/cartpole_clean_deepseek_seed0.yaml"

  echo "============================================================"
  echo "[Hardened clean smoke] CartPole seed0"
  echo "workspace=${WORKSPACE}"
  echo "config=${CONFIG}"
  echo "============================================================"

  rm -rf "$WORKSPACE"
  python run_mvp.py --config "$CONFIG"
  python scripts/audit_clean_run.py "$WORKSPACE"
}

run_lunar() {
  WORKSPACE="runs/clean_lunar_lander_deepseek_seed0_g2p2_t30k"
  CONFIG="mvp/configs/lunar_lander_clean_deepseek_seed0.yaml"

  echo "============================================================"
  echo "[Hardened clean smoke] LunarLander seed0"
  echo "workspace=${WORKSPACE}"
  echo "config=${CONFIG}"
  echo "============================================================"

  rm -rf "$WORKSPACE"
  python run_mvp.py --config "$CONFIG"
  python scripts/audit_clean_run.py "$WORKSPACE"
}

case "$MODE" in
  cartpole)
    run_cartpole
    ;;
  lunar)
    run_lunar
    ;;
  both)
    run_cartpole
    run_lunar
    ;;
  *)
    echo "Usage: bash scripts/run_clean_hardened_single_seed.sh [cartpole|lunar|both]"
    exit 2
    ;;
esac
SH

chmod +x scripts/run_clean_hardened_single_seed.sh

echo "[5/6] syntax check..."
python -m py_compile \
  mvp/clean_feedback.py \
  mvp/agents.py

echo "[6/6] quick static semantic check..."
python - <<'PY'
from mvp.clean_feedback import build_clean_reflection
from mvp.semantic_audit import audit_semantic_text_bundle

text = build_clean_reflection([
    {
        "candidate_id": "g0_c0",
        "status": "ok",
        "selection_score": 1.0,
        "hidden_eval_return": 1.0,
        "train_mean_return": 0.5,
        "repair_attempts": 0,
        "repair_success": False,
        "identity_warning_count": 0,
        "semantic_term_warning_count": 0,
    }
])
audit = audit_semantic_text_bundle({"reflection": text})
print(audit)
assert audit["identity_warning_count"] == 0
assert audit["semantic_term_warning_count"] == 0
PY

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Recommended next:"
echo "  bash scripts/run_clean_hardened_single_seed.sh cartpole"
echo ""
echo "If clean, then:"
echo "  bash scripts/run_clean_hardened_single_seed.sh lunar"
echo ""
echo "Only after both single-seed smokes are clean, consider multiseed."
