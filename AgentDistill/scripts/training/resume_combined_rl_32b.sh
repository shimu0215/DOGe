#!/bin/bash
# resume_combined_rl_32b.sh — Resume Combined-RL from checkpoint-step200
# Branch: claude/可多拉
#
# Resumes from checkpoint-step200.
# STABILITY FIX: lr reduced from 3e-5 → 1e-5 to address the loss spike observed
# at step 220 (loss=-34, r_sens_mean=-8.4) and step 240 (loss=-106).
# lambda_sens and lambda_div_obs kept the same (0.3 and 1.0).
# Runs for 800 more steps (global_step 200 → 1000).

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
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data_processor/math_dataset/test/math_500_20250414.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/training_outputs/qwen3-32B/combined_rl}"
ACCEL_CONFIG="$ROOT_DIR/exps_research/mp_configs/accel_ds3_4gpu.yaml"

# ---- Resume from checkpoint ----
RESUME_CKPT="${RESUME_CKPT:-$OUTPUT_DIR/checkpoint-step200}"
INITIAL_STEP="${INITIAL_STEP:-200}"
MAX_STEPS="${MAX_STEPS:-1000}"   # run until step 1000 total

# ---- Hyperparameters (aggressive, but lr reduced 3e-5→1e-5 for stability) ----
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
SEED_RANGE_START="${SEED_RANGE_START:-42}"
SEED_RANGE_END="${SEED_RANGE_END:-57}"
LAMBDA_SENS="${LAMBDA_SENS:-0.3}"
LAMBDA_DIV_OBS="${LAMBDA_DIV_OBS:-1.0}"
LR="${LR:-1e-5}"          # was 3e-5; reduced for stability
KL_COEFF="${KL_COEFF:-0.001}"
LORA_R="${LORA_R:-32}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
N_TRAJS="${N_TRAJS:-8}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-200}"

mkdir -p logs "$OUTPUT_DIR"

RUN_LOG="$OUTPUT_DIR/run_combined_rl_resume.log"

echo "=== Combined-RL 32B RESUME from step $INITIAL_STEP (lr=1e-5, stabilised) ===" | tee "$RUN_LOG"
echo "  checkpoint:      $RESUME_CKPT"    | tee -a "$RUN_LOG"
echo "  initial_step:    $INITIAL_STEP"   | tee -a "$RUN_LOG"
echo "  max_steps:       $MAX_STEPS"      | tee -a "$RUN_LOG"
echo "  model:           $MODEL_NAME"     | tee -a "$RUN_LOG"
echo "  lambda_sens:     $LAMBDA_SENS"    | tee -a "$RUN_LOG"
echo "  lambda_div_obs:  $LAMBDA_DIV_OBS" | tee -a "$RUN_LOG"
echo "  lr:              $LR (stabilised from 3e-5)" | tee -a "$RUN_LOG"
echo "  kl_coeff:        $KL_COEFF"       | tee -a "$RUN_LOG"
echo "  output_dir:      $OUTPUT_DIR"     | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"

# ---- Preflight check ----
"$PYTHON_BIN" - <<'PY'
import torch, transformers, peft, accelerate
print(f"torch={torch.__version__}  transformers={transformers.__version__}")
print(f"peft={peft.__version__}  accelerate={accelerate.__version__}")
print(f"GPUs: {torch.cuda.device_count()}")
PY

# ---- Launch ----
"$ACCELERATE_BIN" launch \
    --config_file "$ACCEL_CONFIG" \
    -m exps_research.rl_training.train_combined_rl \
    --trajectory_dir    "$TRAJ_DIR" \
    --seed_range        "$SEED_RANGE_START" "$SEED_RANGE_END" \
    --data_path         "$DATA_PATH" \
    --model_name        "$MODEL_NAME" \
    --resume_from_checkpoint "$RESUME_CKPT" \
    --initial_step      "$INITIAL_STEP" \
    --lambda_sens       "$LAMBDA_SENS" \
    --lambda_div_obs    "$LAMBDA_DIV_OBS" \
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
    --log_every         10 \
    2>&1 | tee -a "$RUN_LOG"

echo "=== Combined-RL resume complete ===" | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"
