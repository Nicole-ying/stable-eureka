#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="runs/clean_lunar_lander_ollama_g2p2_t30k"

rm -rf "$WORKSPACE"

python run_mvp.py \
  --config mvp/configs/lunar_lander_clean_ollama_small.yaml

python scripts/audit_clean_run.py "$WORKSPACE"
