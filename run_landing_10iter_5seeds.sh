#!/usr/bin/env bash
set -euo pipefail

BASE_EXP="experiments/eg_rsa_landing_v2_1_source_aware_bootstrap_check"
SCHEMA_PATH="${BASE_EXP}/bootstrap/generated_initial_schema.json"
DIAG_PATH="${BASE_EXP}/bootstrap/generated_diagnostics.yml"

OUT_ROOT="experiments/landing_v2_1_frozen_schema_10iter_5seeds"
CFG_DIR="configs/generated_landing_10iter_5seeds"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
N_ENVS="${N_ENVS:-16}"
N_ITERATIONS="${N_ITERATIONS:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
POSTHOC_EPISODES="${POSTHOC_EPISODES:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

N_GPUS=2  # RTX 3090 × 2

mkdir -p "${OUT_ROOT}"
mkdir -p "${CFG_DIR}"

if [ ! -f "${SCHEMA_PATH}" ]; then
  echo "[ERROR] Missing schema: ${SCHEMA_PATH}"
  exit 1
fi

if [ ! -f "${DIAG_PATH}" ]; then
  echo "[ERROR] Missing diagnostics: ${DIAG_PATH}"
  exit 1
fi

echo "============================================================"
echo "Frozen-schema ${N_ITERATIONS} iter × 5 seeds × ${TOTAL_TIMESTEPS} steps experiment"
echo "SCHEMA_PATH      = ${SCHEMA_PATH}"
echo "DIAG_PATH        = ${DIAG_PATH}"
echo "OUT_ROOT         = ${OUT_ROOT}"
echo "TOTAL_TIMESTEPS  = ${TOTAL_TIMESTEPS}"
echo "N_ITERATIONS     = ${N_ITERATIONS}"
echo "N_ENVS           = ${N_ENVS}"
echo "MAX_PARALLEL     = ${MAX_PARALLEL}"
echo "GPU mapping      = seed % 2  (0/2/4 → GPU 0, 1/3 → GPU 1)"
echo "============================================================"

# Generate per-seed configs
for SEED in 0 1 2 3 4; do
  TRAIN_SEED="${SEED}"
  EVAL_SEED="$((1000 + SEED))"
  POSTHOC_SEED="$((2000 + SEED))"

  CFG_PATH="${CFG_DIR}/eg_rsa_landing_v2_1_10iter_seed${SEED}.yml"
  EXP_DIR="${OUT_ROOT}/seed_${SEED}"

  cat > "${CFG_PATH}" <<YAML
experiment:
  output_dir: ${EXP_DIR}

environment:
  name: landing_runtime_only
  family: landing_control
  gym_id: LunarLander-v3
  kwargs: {}

eg_rsa:
  iterations: ${N_ITERATIONS}

  experiment_mode:
    preset: full

  schema_source:
    type: manual
    initial_schema_path: ${SCHEMA_PATH}

  diagnostic_spec_path: ${DIAG_PATH}

  task_description_inline: >
    Control a lander to approach the landing zone, reduce velocity, keep attitude stable,
    use thrust efficiently, and achieve safe two-leg contact.

  edit_agent:
    backend: deepseek
    model: deepseek-v4-pro
    credential_env: DEEPSEEK_API_KEY
    temperature: 0.0
    timeout: 300

memory:
  top_k: 3
  lesson_top_k: 8
  outcome_lesson_top_k: 8

hack_detector:
  dominance_threshold: 0.7
  event_toggle_threshold: 6
  low_success_threshold: 0.2

candidate_evaluator:
  min_event_trigger_rate: 0.001
  min_metric_variation: 0.0001
  min_metric_active_rate: 0.01
  reject_zero_signal: true
  hard_filter: false
  advisory_only: true

edit_gate:
  max_edits_per_iteration: 3
  min_target_ratio: 0.02
  min_target_trigger_rate: 0.01

outcome_acceptor:
  min_task_improvement: 0.02
  max_task_drop: 0.05
  min_semantic_improvement: 0.05
  max_semantic_drop: 0.05
  min_hack_improvement: 0.1
  max_hack_increase: 0.05

agent_action_controller:
  active: false
  advisory_only: true

scale_audit:
  active: true
  max_dense_to_terminal_ratio: 0.25
  horizon: 1000
  repair_enabled: false
  hard_block: false
  advisory_only: true

behavior_risk_audit:
  active: true
  block_medium: false
  weak_success_evidence: 0.5
  weak_stability_evidence: 0.5
  medium_risk_budget_when_weak_success: 2
  block_medium_under_weak_success: false
  hard_block: false
  advisory_only: true

rl:
  algo: ppo
  algo_params:
    policy: MlpPolicy
    learning_rate: 0.001
    n_steps: 1024
    batch_size: 64
    n_epochs: 4
    gamma: 0.999
    gae_lambda: 0.98
    clip_range: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    policy_kwargs:
      net_arch:
        pi: [64, 64]
        vf: [64, 64]

  training:
    total_timesteps: ${TOTAL_TIMESTEPS}
    seed: ${TRAIN_SEED}
    device: cuda
    verbose: 1
    n_envs: ${N_ENVS}

  evaluation:
    seed: ${EVAL_SEED}
    num_episodes: ${EVAL_EPISODES}

posthoc_eval:
  enabled: true
  seed: ${POSTHOC_SEED}
  num_episodes: ${POSTHOC_EPISODES}
YAML
done

echo "Config files generated under: ${CFG_DIR}"
echo "============================================================"

run_one_seed() {
  local seed="$1"
  local gpu="$2"
  local cfg="${CFG_DIR}/eg_rsa_landing_v2_1_10iter_seed${seed}.yml"
  local log="${OUT_ROOT}/seed_${seed}.log"

  echo "============================================================"
  echo "[START] seed=${seed}  GPU=${gpu}"
  echo "config = ${cfg}"
  echo "log    = ${log}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" python train_eg_rsa.py --config "${cfg}" > "${log}" 2>&1

  echo "============================================================"
  echo "[DONE] seed=${seed}  GPU=${gpu}"
  echo "log = ${log}"
  echo "============================================================"
}

# GPU-queue dispatch: each GPU runs its seeds sequentially.
# GPU 0 seeds (0, 2, 4) and GPU 1 seeds (1, 3) run in parallel across GPUs.
declare -a GPU0_SEEDS=()
declare -a GPU1_SEEDS=()

for SEED in 0 1 2 3 4; do
  if [ "$((SEED % 2))" -eq 0 ]; then
    GPU0_SEEDS+=("${SEED}")
  else
    GPU1_SEEDS+=("${SEED}")
  fi
done

echo "GPU 0 queue: ${GPU0_SEEDS[*]}"
echo "GPU 1 queue: ${GPU1_SEEDS[*]}"
echo "============================================================"

# Launch GPU 0 queue in background
(
  for SEED in "${GPU0_SEEDS[@]}"; do
    run_one_seed "${SEED}" 0
  done
) &
PID_GPU0=$!

# Launch GPU 1 queue in background
(
  for SEED in "${GPU1_SEEDS[@]}"; do
    run_one_seed "${SEED}" 1
  done
) &
PID_GPU1=$!

wait $PID_GPU0 $PID_GPU1

echo "============================================================"
echo "All 5 seeds finished. Generating summary..."
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/landing_v2_1_frozen_schema_10iter_5seeds")
rows = []

for seed_dir in sorted(root.glob("seed_*")):
    seed = int(seed_dir.name.split("_")[-1])
    iters = sorted(seed_dir.glob("iteration_*"))
    if not iters:
        rows.append({"seed": seed, "error": "no iterations found"})
        continue

    last_iter = iters[-1]

    posthoc_path = last_iter / "posthoc_eval.json"
    traj_path   = last_iter / "trajectory_inspection.json"
    diag_path   = last_iter / "diagnostic_report.json"

    row = {
        "seed": seed,
        "last_iteration": last_iter.name,
        "total_iterations": len(iters),
        "posthoc_return_mean": None,
        "posthoc_return_std": None,
        "posthoc_episode_length_mean": None,
        "success_rate": None,
        "stable_landing_rate_mean": None,
        "safe_contact_rate_mean": None,
        "contact_toggle_mean": None,
        "semantic_score": None,
        "hack_score": None,
        "failure_modes": [],
    }

    if posthoc_path.exists():
        data = json.loads(posthoc_path.read_text(encoding="utf-8"))
        row["posthoc_return_mean"] = data.get("return_mean")
        row["posthoc_return_std"] = data.get("return_std")
        row["posthoc_episode_length_mean"] = data.get("episode_length_mean")

    if traj_path.exists():
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        row["success_rate"] = data.get("success_rate")
        row["stable_landing_rate_mean"] = data.get("stable_landing_rate_mean")
        row["safe_contact_rate_mean"] = data.get("safe_contact_rate_mean")
        row["contact_toggle_mean"] = data.get("contact_toggle_mean")

    if diag_path.exists():
        data = json.loads(diag_path.read_text(encoding="utf-8"))
        row["semantic_score"] = data.get("semantic_outcome", {}).get("semantic_score")
        row["hack_score"] = data.get("diagnostics", {}).get("hack_score")
        row["failure_modes"] = data.get("diagnostics", {}).get("failure_modes", [])

    rows.append(row)

summary_path = root / "summary_5seeds.json"
summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(rows, indent=2, ensure_ascii=False))
print(f"Saved summary to: {summary_path}")
PY

echo "============================================================"
echo "All done."
echo "Results under: ${OUT_ROOT}"
echo "Summary: ${OUT_ROOT}/summary_5seeds.json"
echo "============================================================"
