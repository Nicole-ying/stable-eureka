#!/usr/bin/env bash
# =============================================================================
# run_rewardspec_lunar_g16_t2m_2seeds.sh
#
# Purpose:
#   Launch two independent EG-RSA RewardSpec single-chain long runs.
#
# Runs:
#   seed0: 16 generations × 1 candidate × 2M PPO steps, GPU 0
#   seed1: 16 generations × 1 candidate × 2M PPO steps, GPU 1
#
# Notes:
#   - RewardSpec/LLM never sees official reward.
#   - private_eval_return / eval_hidden_return is the posthoc official reward eval.
#   - Workspaces are separated by seed.
#   - Logs and pid files are saved under runs/longrun_logs/.
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="runs/longrun_logs"
mkdir -p "$LOG_DIR"

CFG0="mvp/configs/eg_rsa_lunar_deepseek_seed0_singlechain_g16_t2m.yaml"
CFG1="mvp/configs/eg_rsa_lunar_deepseek_seed1_singlechain_g16_t2m.yaml"

RUN0="runs/eg_rsa_lunar_deepseek_seed0_singlechain_g16_t2m"
RUN1="runs/eg_rsa_lunar_deepseek_seed1_singlechain_g16_t2m"

if [ ! -f "$CFG0" ]; then
  echo "ERROR: missing config $CFG0"
  exit 1
fi

if [ ! -f "$CFG1" ]; then
  echo "ERROR: missing config $CFG1"
  exit 1
fi

if [ -d "$RUN0" ] || [ -d "$RUN1" ]; then
  echo "ERROR: one or both run directories already exist."
  echo "  $RUN0"
  echo "  $RUN1"
  echo "Move or remove them first if you want a clean run."
  exit 1
fi

echo "[check] Python syntax..."
python -m py_compile mvp/*.py scripts/check_run_quality.py

echo "[launch] seed0 on GPU0..."
CUDA_VISIBLE_DEVICES=0 nohup python run_mvp.py --config "$CFG0" \
  > "$LOG_DIR/seed0_g16_t2m.log" 2>&1 &
PID0=$!
echo "$PID0" > "$LOG_DIR/seed0_g16_t2m.pid"

echo "[launch] seed1 on GPU1..."
CUDA_VISIBLE_DEVICES=1 nohup python run_mvp.py --config "$CFG1" \
  > "$LOG_DIR/seed1_g16_t2m.log" 2>&1 &
PID1=$!
echo "$PID1" > "$LOG_DIR/seed1_g16_t2m.pid"

echo ""
echo "Launched long runs:"
echo "  seed0 PID=$PID0 GPU=0 config=$CFG0"
echo "  seed1 PID=$PID1 GPU=1 config=$CFG1"
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/seed0_g16_t2m.log"
echo "  tail -f $LOG_DIR/seed1_g16_t2m.log"
echo ""
echo "PID files:"
echo "  $LOG_DIR/seed0_g16_t2m.pid"
echo "  $LOG_DIR/seed1_g16_t2m.pid"
echo ""
echo "Check status:"
echo "  ps -p $PID0 -o pid,etime,cmd"
echo "  ps -p $PID1 -o pid,etime,cmd"
echo ""
echo "After completion:"
echo "  python scripts/check_run_quality.py $RUN0"
echo "  python scripts/check_run_quality.py $RUN1"
