#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="runs/mvp_cartpole_mock"

rm -rf "$WORKSPACE"

python run_mvp.py \
  --config mvp/configs/cartpole_mock.yaml \
  --provider mock \
  --timesteps 2000

python scripts/audit_clean_run.py "$WORKSPACE"
