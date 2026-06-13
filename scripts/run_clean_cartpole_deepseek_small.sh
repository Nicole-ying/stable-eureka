#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="runs/clean_cartpole_deepseek_g2p2_t8k"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set."
  echo "Please run:"
  echo "  export DEEPSEEK_API_KEY='your_key_here'"
  exit 1
fi

rm -rf "$WORKSPACE"

python run_mvp.py \
  --config mvp/configs/cartpole_clean_deepseek_small.yaml

python scripts/audit_clean_run.py "$WORKSPACE"
