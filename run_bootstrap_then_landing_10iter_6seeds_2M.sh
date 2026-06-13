#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_bootstrap_then_landing_10iter_6seeds_2M.sh
#
# 目标：
#   1. 先重新跑 source-aware bootstrap/preflight，生成：
#        - generated_initial_schema.json
#        - generated_diagnostics.yml
#        - generated_primitive_interface.json
#   2. 再启动 6 seeds × 10 iter × 2M steps 长实验。
#
# 核心原则：
#   bootstrap 只做一次。
#   所有 seeds 复用同一份 bootstrap 产物。
# =============================================================================

BOOTSTRAP_CONFIG="configs/eg_rsa_landing_v2_1_source_aware_bootstrap_check.yml"
BASE_EXP="experiments/eg_rsa_landing_v2_1_source_aware_bootstrap_check"

SCHEMA_PATH="${BASE_EXP}/bootstrap/generated_initial_schema.json"
DIAG_PATH="${BASE_EXP}/bootstrap/generated_diagnostics.yml"
INTERFACE_PATH="${BASE_EXP}/interface/generated_primitive_interface.json"

LONG_SCRIPT="./run_landing_10iter_6seeds_2M.sh"
LONG_EXP="experiments/landing_v2_1_10iter_6seeds_2M"

# 默认重新 bootstrap。因为我们刚修了 diagnostics 生成逻辑，必须重跑。
FORCE_BOOTSTRAP="${FORCE_BOOTSTRAP:-1}"

# 默认备份旧长实验目录，避免覆盖之前中断的实验。
BACKUP_OLD_LONG="${BACKUP_OLD_LONG:-1}"

echo "============================================================"
echo "Bootstrap + long experiment pipeline"
echo "BOOTSTRAP_CONFIG = ${BOOTSTRAP_CONFIG}"
echo "BASE_EXP         = ${BASE_EXP}"
echo "SCHEMA_PATH      = ${SCHEMA_PATH}"
echo "DIAG_PATH        = ${DIAG_PATH}"
echo "INTERFACE_PATH   = ${INTERFACE_PATH}"
echo "LONG_SCRIPT      = ${LONG_SCRIPT}"
echo "LONG_EXP         = ${LONG_EXP}"
echo "FORCE_BOOTSTRAP  = ${FORCE_BOOTSTRAP}"
echo "BACKUP_OLD_LONG  = ${BACKUP_OLD_LONG}"
echo "TOTAL_TIMESTEPS  = ${TOTAL_TIMESTEPS:-2000000}"
echo "N_ENVS           = ${N_ENVS:-16}"
echo "N_ITERATIONS     = ${N_ITERATIONS:-10}"
echo "============================================================"

test -f train_eg_rsa.py
test -f "${BOOTSTRAP_CONFIG}"
test -f "${LONG_SCRIPT}"

if [ "${FORCE_BOOTSTRAP}" = "1" ]; then
  if [ -d "${BASE_EXP}" ]; then
    BK="${BASE_EXP}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "[BOOTSTRAP] Backup old bootstrap dir:"
    echo "  ${BASE_EXP} -> ${BK}"
    mv "${BASE_EXP}" "${BK}"
  fi

  echo "[BOOTSTRAP] Running source-aware bootstrap/preflight..."
  python train_eg_rsa.py --config "${BOOTSTRAP_CONFIG}"
else
  echo "[BOOTSTRAP] FORCE_BOOTSTRAP=0, reusing existing bootstrap artifacts."
fi

echo "[CHECK] Verify bootstrap artifacts..."
if [ ! -f "${SCHEMA_PATH}" ]; then
  echo "[ERROR] Missing schema: ${SCHEMA_PATH}"
  exit 1
fi

if [ ! -f "${DIAG_PATH}" ]; then
  echo "[ERROR] Missing diagnostics: ${DIAG_PATH}"
  exit 1
fi

if [ ! -f "${INTERFACE_PATH}" ]; then
  echo "[ERROR] Missing primitive interface: ${INTERFACE_PATH}"
  exit 1
fi

echo "[CHECK] Verify diagnostics contains landing semantic probes..."
REQ_KEYS=(
  "both_contact"
  "safe_contact"
  "stable_landing_condition"
  "success"
  "landing_quality"
)

for key in "${REQ_KEYS[@]}"; do
  if ! grep -q "${key}" "${DIAG_PATH}"; then
    echo "[ERROR] New diagnostics is missing required key: ${key}"
    echo "        Check _build_runtime_spec_from_primitive_interface patch."
    exit 1
  fi
done

echo "[CHECK] Diagnostics semantic probes OK."

echo "[CHECK] Verify long script has primitive_interface_path..."
if ! grep -q 'primitive_interface_path: ${INTERFACE_PATH}' "${LONG_SCRIPT}"; then
  echo "[ERROR] ${LONG_SCRIPT} does not include primitive_interface_path."
  echo "        Please rerun patch_eg_rsa_edit_contract_fix.sh first."
  exit 1
fi

if [ "${BACKUP_OLD_LONG}" = "1" ] && [ -d "${LONG_EXP}" ]; then
  BK="${LONG_EXP}.bak_$(date +%Y%m%d_%H%M%S)"
  echo "[LONG] Backup old long experiment dir:"
  echo "  ${LONG_EXP} -> ${BK}"
  mv "${LONG_EXP}" "${BK}"
fi

mkdir -p "${LONG_EXP}"

echo "============================================================"
echo "[LONG] Starting long experiment..."
echo "This will call: ${LONG_SCRIPT}"
echo "============================================================"

env \
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}" \
  N_ENVS="${N_ENVS:-16}" \
  N_ITERATIONS="${N_ITERATIONS:-10}" \
  "${LONG_SCRIPT}"

echo "============================================================"
echo "Bootstrap + long experiment finished."
echo "Results under: ${LONG_EXP}"
echo "============================================================"
