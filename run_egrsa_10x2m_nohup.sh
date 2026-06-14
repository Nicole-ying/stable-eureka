#!/bin/bash
set -u

mkdir -p logs

RUN_NAME="egrsa_10x2m"
LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/${RUN_NAME}.pid"

CONFIG="eg_rsa/single_chain/configs/lunar_lander_10x2m.yaml"

echo "============================================================" | tee -a "$LOG_FILE"
echo "[START] $(date)" | tee -a "$LOG_FILE"
echo "[PWD]   $(pwd)" | tee -a "$LOG_FILE"
echo "[HOST]  $(hostname)" | tee -a "$LOG_FILE"
echo "[CONFIG] $CONFIG" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "[GIT] branch:" | tee -a "$LOG_FILE"
git branch --show-current 2>&1 | tee -a "$LOG_FILE"

echo "[GIT] latest commit:" | tee -a "$LOG_FILE"
git log --oneline -1 2>&1 | tee -a "$LOG_FILE"

echo "[OLLAMA] model check:" | tee -a "$LOG_FILE"
ollama list | grep "qwen3.5:27b-eg-rsa-32k" 2>&1 | tee -a "$LOG_FILE"

echo "[GPU] initial nvidia-smi:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE"

echo "============================================================" | tee -a "$LOG_FILE"
echo "[RUN] python -u -m eg_rsa.run_single_chain_iterative --config $CONFIG" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 后台心跳：每 10 分钟写一次实验进度到 log
(
  while true; do
    sleep 600
    echo "" 
    echo "==================== [HEARTBEAT] $(date) ===================="
    echo "[PROCESS]"
    ps -fp "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || true

    echo ""
    echo "[GPU]"
    nvidia-smi 2>/dev/null || true

    echo ""
    echo "[LATEST EXPERIMENT DIR]"
    LATEST_DIR=$(ls -td experiments/eg_rsa_single_chain/* 2>/dev/null | head -n 1)
    echo "$LATEST_DIR"

    if [ -n "${LATEST_DIR:-}" ] && [ -d "$LATEST_DIR" ]; then
      echo ""
      echo "[TOP FILES]"
      find "$LATEST_DIR" -maxdepth 4 -type f \
        \( -name "final_eval.json" -o -name "reward_trace.json" -o -name "iteration_evidence.json" -o -name "reflection_decision.json" -o -name "revised_reward_schema_and_code.json" -o -name "memory_transition.json" \) \
        -printf "%TY-%Tm-%Td %TH:%TM:%TS %p\n" 2>/dev/null | sort | tail -n 30
    fi

    echo "=============================================================="
    echo ""
  done
) >> "$LOG_FILE" 2>&1 &
MONITOR_PID=$!

# 主训练进程
python -u -m eg_rsa.run_single_chain_iterative \
  --config "$CONFIG" >> "$LOG_FILE" 2>&1 &

MAIN_PID=$!
echo "$MAIN_PID" > "$PID_FILE"

echo "[MAIN_PID] $MAIN_PID" | tee -a "$LOG_FILE"
echo "[MONITOR_PID] $MONITOR_PID" | tee -a "$LOG_FILE"
echo "[LOG_FILE] $LOG_FILE" | tee -a "$LOG_FILE"

wait "$MAIN_PID"
EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "[END] $(date)" | tee -a "$LOG_FILE"
echo "[EXIT_CODE] $EXIT_CODE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

kill "$MONITOR_PID" 2>/dev/null || true

exit "$EXIT_CODE"
