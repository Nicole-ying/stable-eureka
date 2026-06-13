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
