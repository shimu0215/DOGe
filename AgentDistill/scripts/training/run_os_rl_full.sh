#!/bin/bash
# run_os_rl_full.sh — OS-RL full MATH-500: all 500 questions, seeds 42-56
# Branch: claude/可多拉
#
# Difference from run_os_rl_pilot.sh:
#   - trajectory_dir points directly to the evaluations dir (all 500 questions)
#   - No prepare_pilot_data step needed
#   - Output dirs: os_rl_full_sft_init / os_rl_full_v2 (set via OUTPUT_DIR)
#   - All other hyperparams identical to pilot
#
# Usage:
#   # Without SFT init (v2):
#   OUTPUT_DIR=.../os_rl_full_v2 bash scripts/training/run_os_rl_full.sh
#
#   # With SFT init:
#   OUTPUT_DIR=.../os_rl_full_sft_init \
#   RESUME_FROM_CHECKPOINT=.../agent_baseline_2epochs_... \
#   bash scripts/training/run_os_rl_full.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- Environment ----
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
export PATH="$CONDA_ENV_PREFIX/bin:${PATH:-}"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wzhao20/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="$CONDA_ENV_PREFIX/bin/python"
ACCELERATE_BIN="$CONDA_ENV_PREFIX/bin/accelerate"

# ---- Paths ----
TRAJ_DIR="${TRAJ_DIR:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test/evaluations}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/training_outputs/qwen3-32B/os_rl_full_v2}"
ACCEL_CONFIG="$ROOT_DIR/exps_research/mp_configs/accel_ds3_4gpu.yaml"

# ---- Parameters (same as pilot) ----
SEED_RANGE_START="${SEED_RANGE_START:-42}"
SEED_RANGE_END="${SEED_RANGE_END:-57}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
LAMBDA_SENS="${LAMBDA_SENS:-0.1}"
LR="${LR:-1e-5}"
KL_COEFF="${KL_COEFF:-0.01}"
LORA_R="${LORA_R:-32}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-150}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
N_TRAJS="${N_TRAJS:-8}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-50}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INITIAL_STEP="${INITIAL_STEP:-0}"

mkdir -p "$OUTPUT_DIR"

RUN_LOG="$OUTPUT_DIR/run_os_rl_full.log"

echo "=== OS-RL FULL MATH-500 ===" | tee "$RUN_LOG"
date | tee -a "$RUN_LOG"
echo "  traj_dir:     $TRAJ_DIR"     | tee -a "$RUN_LOG"
echo "  output_dir:   $OUTPUT_DIR"   | tee -a "$RUN_LOG"
echo "  seed_range:   $SEED_RANGE_START-$SEED_RANGE_END" | tee -a "$RUN_LOG"
echo "  resume_ckpt:  ${RESUME_FROM_CHECKPOINT:-(none)}" | tee -a "$RUN_LOG"

# ---- Preflight check ----
"$PYTHON_BIN" - <<'PY'
import torch, transformers, peft, accelerate
print(f"torch={torch.__version__}  transformers={transformers.__version__}")
print(f"peft={peft.__version__}  accelerate={accelerate.__version__}")
print(f"GPUs: {torch.cuda.device_count()}")
PY

# ---- Launch OS-RL training ----
echo "" | tee -a "$RUN_LOG"
echo "Launching OS-RL full training..." | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"

RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" --initial_step "$INITIAL_STEP")
fi

"$ACCELERATE_BIN" launch \
    --config_file "$ACCEL_CONFIG" \
    -m exps_research.rl_training.train_os_rl \
    --trajectory_dir    "$TRAJ_DIR" \
    --seed_range        "$SEED_RANGE_START" "$SEED_RANGE_END" \
    --model_name        "$MODEL_NAME" \
    --lambda_sens       "$LAMBDA_SENS" \
    --lr                "$LR" \
    --kl_coeff          "$KL_COEFF" \
    --lora_r            "$LORA_R" \
    --max_length        "$MAX_LENGTH" \
    --num_epochs        "$NUM_EPOCHS" \
    --max_steps         "$MAX_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --n_trajs_per_question "$N_TRAJS" \
    --resample_every    "$RESAMPLE_EVERY" \
    --output_dir        "$OUTPUT_DIR" \
    --log_every         5 \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$RUN_LOG"

echo "=== OS-RL full MATH-500 complete ===" | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"
