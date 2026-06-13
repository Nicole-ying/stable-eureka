#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="runs/clean_cartpole_ollama_g2p2_t8k"

rm -rf "$WORKSPACE"

python run_mvp.py \
  --config mvp/configs/cartpole_clean_ollama_small.yaml

python scripts/audit_clean_run.py "$WORKSPACE"
