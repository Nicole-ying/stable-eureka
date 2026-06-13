#!/usr/bin/env bash
set -euo pipefail

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set."
  echo "Please run:"
  echo "  export DEEPSEEK_API_KEY='your_key_here'"
  exit 1
fi

for SEED in 0 1 2; do
  WORKSPACE="runs/clean_lunar_lander_deepseek_seed${SEED}_g2p2_t30k"
  CONFIG="mvp/configs/lunar_lander_clean_deepseek_seed${SEED}.yaml"

  echo "============================================================"
  echo "[LunarLander clean DeepSeek] seed=${SEED}"
  echo "workspace=${WORKSPACE}"
  echo "config=${CONFIG}"
  echo "============================================================"

  rm -rf "$WORKSPACE"

  python run_mvp.py --config "$CONFIG"
  python scripts/audit_clean_run.py "$WORKSPACE"
done
