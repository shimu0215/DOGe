#!/bin/bash
# OS-RL offline GRPO training on pre-collected MATH-500 trajectories (seeds 42-53).
# Runs on a single process with device_map=auto across 4×A100 80GB.
#
# Usage:
#   bash scripts/training/run_osrl_offline.sh
#
# Resume after GPU allocation expires:
#   RESUME=/path/to/checkpoints/iter0050 bash scripts/training/run_osrl_offline.sh
#
# Key env-var overrides:
#   MODEL_NAME, DATA_DIR, SEED_START, SEED_END, OUTPUT_DIR,
#   LAMBDA_SENSITIVITY, NUM_ITERATIONS, BATCH_SIZE, G_PER_PROBLEM,
#   LR, KL_COEF, LORA_RANK, MAX_SEQ_LEN, SAVE_EVERY, RESUME

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
source /home/wzhao20/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/.bashrc 2>/dev/null || true
conda activate agents

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ #
# Hyperparameters (override via env vars)
# ------------------------------------------------------------------ #
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"

# Pre-collected trajectories (seeds 42-53)
DATA_DIR="${DATA_DIR:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test}"
SEED_START="${SEED_START:-42}"
SEED_END="${SEED_END:-53}"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/wzhao20/AKDA2-vjk/AgentDistill/training_outputs/osrl/offline_run1}"

LAMBDA_SENSITIVITY="${LAMBDA_SENSITIVITY:-0.1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"        # problems per iteration
G_PER_PROBLEM="${G_PER_PROBLEM:-4}"  # rollouts per problem (keep it small for speed)

LR="${LR:-2e-6}"
KL_COEF="${KL_COEF:-0.01}"
CLIP_EPS="${CLIP_EPS:-0.2}"
LORA_RANK="${LORA_RANK:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-6144}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SEED="${SEED:-42}"

RESUME="${RESUME:-}"

# ------------------------------------------------------------------ #
# Launch
# ------------------------------------------------------------------ #
mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " OS-RL Offline GRPO Training"
echo "  model          : $MODEL_NAME"
echo "  data_dir       : $DATA_DIR"
echo "  seeds          : $SEED_START - $SEED_END"
echo "  output_dir     : $OUTPUT_DIR"
echo "  lambda_sens    : $LAMBDA_SENSITIVITY"
echo "  num_iterations : $NUM_ITERATIONS"
echo "  batch_size     : $BATCH_SIZE"
echo "  g_per_problem  : $G_PER_PROBLEM"
echo "  lr             : $LR"
echo "  kl_coef        : $KL_COEF"
echo "  lora_rank      : $LORA_RANK"
echo "  max_seq_len    : $MAX_SEQ_LEN"
echo "  save_every     : $SAVE_EVERY"
if [[ -n "$RESUME" ]]; then
    echo "  resume         : $RESUME"
fi
echo "============================================="

RESUME_ARG=""
if [[ -n "$RESUME" ]]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME"
fi

python -u -m exps_research.rl.osrl_offline_train \
    --model_name              "$MODEL_NAME"           \
    --data_dir                "$DATA_DIR"             \
    --seed_start              "$SEED_START"           \
    --seed_end                "$SEED_END"             \
    --output_dir              "$OUTPUT_DIR"           \
    --lambda_sensitivity      "$LAMBDA_SENSITIVITY"   \
    --num_iterations          "$NUM_ITERATIONS"       \
    --batch_size              "$BATCH_SIZE"           \
    --g_per_problem           "$G_PER_PROBLEM"        \
    --lr                      "$LR"                   \
    --kl_coef                 "$KL_COEF"              \
    --clip_eps                "$CLIP_EPS"             \
    --lora_rank               "$LORA_RANK"            \
    --max_seq_length          "$MAX_SEQ_LEN"          \
    --grad_accum_steps        "$GRAD_ACCUM"           \
    --save_every              "$SAVE_EVERY"           \
    --seed                    "$SEED"                 \
    $RESUME_ARG \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

echo "Done. Results in $OUTPUT_DIR"
