#!/bin/bash
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set."
  echo "  export DEEPSEEK_API_KEY='your_key_here'"
  exit 1
fi
LOG="/home/utseus22/stable-eureka-nicole/logs/egrsa_3x300k_smoke_ds_$(date +%Y%m%d_%H%M%S).log"
cd /home/utseus22/stable-eureka-nicole
exec >> "$LOG" 2>&1
echo "=== Smoke test started at $(date) ==="
conda run -n stable-eureka --no-capture-output python -m eg_rsa.run_single_chain_iterative \
  --config eg_rsa/single_chain/configs/lunar_lander_3x300k_smoke.yaml
echo "=== Done at $(date) with exit code $? ==="
