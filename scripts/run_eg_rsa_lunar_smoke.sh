#!/usr/bin/env bash
set -euo pipefail

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY first}"

rm -rf runs/eg_rsa_lunar_deepseek_smoke

python run_mvp.py \
  --config mvp/configs/eg_rsa_lunar_deepseek_smoke.yaml
