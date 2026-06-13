#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_clean_cartpole_deepseek_multiseed.sh
bash scripts/run_clean_lunar_deepseek_multiseed.sh

python scripts/summarize_clean_multiseed.py \
  runs/clean_cartpole_deepseek_seed0_g2p2_t8k \
  runs/clean_cartpole_deepseek_seed1_g2p2_t8k \
  runs/clean_cartpole_deepseek_seed2_g2p2_t8k \
  runs/clean_lunar_lander_deepseek_seed0_g2p2_t30k \
  runs/clean_lunar_lander_deepseek_seed1_g2p2_t30k \
  runs/clean_lunar_lander_deepseek_seed2_g2p2_t30k \
  --out runs/clean_multiseed_summary.csv
