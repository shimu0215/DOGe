#!/bin/bash
# OS-RL pilot: offline GRPO on pre-collected Qwen3-32B MATH-500 trajectories.
# Runs entirely in AKDA2-vjk worktree — does NOT touch main branch.
#
# Usage:
#   bash scripts/training/run_osrl_pilot.sh
#
# Validates:
#   - sensitivity reward computation on real trajectories
#   - GRPO backward pass with LoRA on 4×A100 80G
#   - GPU memory budget
#   - reward stats (R_task, R_sensitivity)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Activate the conda environment that has peft/transformers/torch
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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"

# Pre-collected Qwen3-32B MATH-500 seed=57 trajectories
ROLLOUT_DIR="${ROLLOUT_DIR:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher}"
PATTERN="${PATTERN:-math_500_20250414_test/Qwen3-32B_temp=0.7_seed=57_type=agent_steps=5_python_only_python_only_seed57.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/wzhao20/AKDA2-vjk/AgentDistill/training_outputs/osrl/pilot}"

N_PROBLEMS="${N_PROBLEMS:-16}"
LAMBDA_SENSITIVITY="${LAMBDA_SENSITIVITY:-0.1}"
LORA_RANK="${LORA_RANK:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LR="${LR:-1e-5}"
KL_COEF="${KL_COEF:-0.01}"
SEED="${SEED:-42}"
NPROC="${NPROC:-4}"

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " OS-RL Pilot (offline GRPO)"
echo "  model         : $MODEL_NAME"
echo "  rollout_dir   : $ROLLOUT_DIR"
echo "  pattern       : $PATTERN"
echo "  n_problems    : $N_PROBLEMS"
echo "  lambda_sens   : $LAMBDA_SENSITIVITY"
echo "  lora_rank     : $LORA_RANK"
echo "  max_seq_len   : $MAX_SEQ_LEN"
echo "  output_dir    : $OUTPUT_DIR"
echo "  nproc         : $NPROC"
echo "============================================="

torchrun \
    --nproc_per_node="$NPROC" \
    --master_port=29501 \
    -m exps_research.rl.osrl_pilot \
    --model_name          "$MODEL_NAME"         \
    --rollout_dir         "$ROLLOUT_DIR"         \
    --pattern             "$PATTERN"             \
    --output_dir          "$OUTPUT_DIR"          \
    --n_problems          "$N_PROBLEMS"          \
    --lambda_sensitivity  "$LAMBDA_SENSITIVITY"  \
    --lora_rank           "$LORA_RANK"           \
    --max_seq_length      "$MAX_SEQ_LEN"         \
    --learning_rate       "$LR"                  \
    --kl_coef             "$KL_COEF"             \
    --seed                "$SEED"                \
    2>&1 | tee "$OUTPUT_DIR/pilot.log"

echo "Done. Results in $OUTPUT_DIR"
