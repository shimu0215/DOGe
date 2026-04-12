#!/usr/bin/env bash
# run_grpo_online.sh — Launch online GRPO training on Qwen3-14B.
#
# GPU layout:
#   GPU 0    CUDA_VISIBLE_DEVICES=0   — vLLM rollout (tp=1, 14B fits in 80G)
#   GPU 2,3  CUDA_VISIBLE_DEVICES=2,3 — accelerate training (ZeRO-3, LoRA)
#
# All configurable params can be overridden via environment variables.
set -euo pipefail

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # AgentDistill/

QUESTION_JSON="${QUESTION_JSON:-/scratch/wzhao20/AKDA2-canduola/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/training_outputs/qwen3-14B/grpo_online}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"

# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------
K="${K:-8}"                            # trajectories per question per rollout
BATCH_QUESTIONS="${BATCH_QUESTIONS:-4}" # questions per rollout batch
ROLLOUT_EVERY="${ROLLOUT_EVERY:-8}"    # refresh pool every N training steps
ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-5}"
N_TRAJS_PER_Q="${N_TRAJS_PER_Q:-8}"

# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------
LAMBDA_KL="${LAMBDA_KL:-0.1}"          # weight for KL divergence reward

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
MAX_STEPS="${MAX_STEPS:-400}"
LORA_R="${LORA_R:-16}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
LOG_EVERY="${LOG_EVERY:-5}"
SEED="${SEED:-42}"

# --------------------------------------------------------------------------
# Infra: accelerate config and GPU binding for training (GPU 2,3)
# --------------------------------------------------------------------------
ACCEL_CONFIG="$REPO_ROOT/exps_research/mp_configs/accel_ds3_2gpu.yaml"

# Training process sees only GPU 2,3; vLLM subprocess will be given 0,1
export CUDA_VISIBLE_DEVICES=2,3

# Disable NCCL watchdog so training ranks don't die while rank0 waits on
# vLLM rollout (which loads the model and runs inference on GPU 0,1).
export TORCH_NCCL_ENABLE_MONITORING=0

# Offline mode: model weights already on disk
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "========================================================"
echo "  Online GRPO — Qwen3-14B"
echo "  model:           $MODEL_NAME"
echo "  output_dir:      $OUTPUT_DIR"
echo "  question_json:   $QUESTION_JSON"
echo "  K:               $K  (trajectories/question/rollout)"
echo "  batch_questions: $BATCH_QUESTIONS"
echo "  rollout_every:   $ROLLOUT_EVERY  steps"
echo "  lambda_kl:       $LAMBDA_KL"
echo "  max_steps:       $MAX_STEPS"
echo "  checkpoint_every:$CHECKPOINT_EVERY"
echo "  training GPUs:   2,3  |  rollout GPU: 0 (tp=1)"
echo "========================================================"
date

mkdir -p "$OUTPUT_DIR"

# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------
exec accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --main_process_port 29600 \
    -m exps_research.rl_training.train_grpo_online \
        --question_json       "$QUESTION_JSON" \
        --model_name          "$MODEL_NAME" \
        --output_dir          "$OUTPUT_DIR" \
        --lora_r              "$LORA_R" \
        --K                   "$K" \
        --batch_questions     "$BATCH_QUESTIONS" \
        --rollout_every       "$ROLLOUT_EVERY" \
        --rollout_temperature "$ROLLOUT_TEMP" \
        --rollout_top_p       "$ROLLOUT_TOP_P" \
        --max_agent_steps     "$MAX_AGENT_STEPS" \
        --n_trajs_per_question "$N_TRAJS_PER_Q" \
        --lambda_kl           "$LAMBDA_KL" \
        --max_steps           "$MAX_STEPS" \
        --lr                  "$LR" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --checkpoint_every    "$CHECKPOINT_EVERY" \
        --log_every           "$LOG_EVERY" \
        --seed                "$SEED"
