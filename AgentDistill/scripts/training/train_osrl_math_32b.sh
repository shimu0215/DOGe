#!/bin/bash
# OS-RL training launch script for Qwen3-32B on 4x A100 80GB.
#
# Usage:
#   bash scripts/training/train_osrl_math_32b.sh
#
# Key env-var overrides:
#   MODEL_NAME, DATA_PATH, OUTPUT_DIR,
#   LAMBDA_SENSITIVITY, NUM_ROLLOUTS, ROLLOUT_BATCH,
#   NUM_ITERS, LR, KL_COEF, LORA_RANK, VLLM_TP

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

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

# ------------------------------------------------------------------ #
# Hyperparameters (override via env vars)
# ------------------------------------------------------------------ #
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
DATA_PATH="${DATA_PATH:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/wzhao20/AgentDistill/training_outputs/osrl/qwen3-32b-math-osrl}"

LAMBDA_SENSITIVITY="${LAMBDA_SENSITIVITY:-0.1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"         # G per problem
ROLLOUT_BATCH="${ROLLOUT_BATCH:-32}"     # problems per RL iteration
NUM_ITERS="${NUM_ITERS:-100}"

LR="${LR:-1e-5}"
KL_COEF="${KL_COEF:-0.01}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
VLLM_TP="${VLLM_TP:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-10240}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-42}"

RESUME="${RESUME:-}"   # set to checkpoint path to resume

# ------------------------------------------------------------------ #
# Launch
# ------------------------------------------------------------------ #
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "OS-RL Training: Qwen3-32B on MATH-500"
echo "  output_dir        : $OUTPUT_DIR"
echo "  lambda_sensitivity: $LAMBDA_SENSITIVITY"
echo "  num_rollouts/prob : $NUM_ROLLOUTS"
echo "  rollout_batch     : $ROLLOUT_BATCH"
echo "  num_iters         : $NUM_ITERS"
echo "  lr                : $LR"
echo "  kl_coef           : $KL_COEF"
echo "  lora_rank         : $LORA_RANK"
echo "=========================================="

RESUME_ARG=""
if [[ -n "$RESUME" ]]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME"
fi

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m exps_research.rl.train_osrl \
    --model_name        "$MODEL_NAME" \
    --data_path         "$DATA_PATH" \
    --output_dir        "$OUTPUT_DIR" \
    --lambda_sensitivity "$LAMBDA_SENSITIVITY" \
    --num_rollouts_per_problem "$NUM_ROLLOUTS" \
    --rollout_batch_size "$ROLLOUT_BATCH" \
    --num_rl_iterations  "$NUM_ITERS" \
    --learning_rate      "$LR" \
    --kl_coef            "$KL_COEF" \
    --lora_rank          "$LORA_RANK" \
    --lora_alpha         "$LORA_ALPHA" \
    --vllm_tp_size       "$VLLM_TP" \
    --max_seq_length     "$MAX_SEQ_LEN" \
    --save_every_n_iters "$SAVE_EVERY" \
    --seed               "$SEED" \
    $RESUME_ARG \
    2>&1 | tee "$OUTPUT_DIR/train_osrl.log"
