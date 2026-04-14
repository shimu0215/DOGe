#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
export PATH="$CONDA_ENV_PREFIX/bin:${PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wzhao20/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
DATA_PATH="${DATA_PATH:-/scratch/wzhao20/AKDA2/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/wzhao20/AKDA2/AgentDistill/training_outputs/qwen3-14B/agent_online_grpo_math500}"
SEED="${SEED:-42}"
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-2,3}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"
ROLLOUT_API_BASE="${ROLLOUT_API_BASE:-http://127.0.0.1:${ROLLOUT_PORT}/v1}"
NUM_QUESTIONS_PER_SYNC="${NUM_QUESTIONS_PER_SYNC:-16}"
NUM_ROLLOUTS_PER_QUESTION="${NUM_ROLLOUTS_PER_QUESTION:-4}"
MAX_SYNCS="${MAX_SYNCS:-32}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
KL_LAMBDA="${KL_LAMBDA:-0.05}"
SAVE_EVERY_SYNCS="${SAVE_EVERY_SYNCS:-2}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-5}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
RESUME_FROM_ADAPTER="${RESUME_FROM_ADAPTER:-}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$CONDA_ENV_PREFIX/bin/accelerate}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PREFIX/bin/python}"
ROLLOUT_ONLY="${ROLLOUT_ONLY:-0}"
ROLLOUT_LOG="${ROLLOUT_LOG:-$OUTPUT_ROOT/rollout_server.log}"
ROLLOUT_TIMEOUT_SECONDS="${ROLLOUT_TIMEOUT_SECONDS:-1800}"

wait_for_rollout_server() {
  local log_file="$1"
  local timeout_s="$2"
  local waited=0
  until grep -q "Application startup complete" "$log_file" 2>/dev/null; do
    if (( waited >= timeout_s )); then
      echo "Timed out waiting for rollout server startup: $log_file" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

mkdir -p "$OUTPUT_ROOT"

if [[ "$ROLLOUT_ONLY" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="$ROLLOUT_CUDA_VISIBLE_DEVICES"
  printf 'ROLLOUT_CMD: %q %q %q %q %q %q %q %q %q %q %q %q %q %q %q\n' \
    "$PYTHON_BIN" serve_vllm.py --model "$MODEL_NAME" --port "$ROLLOUT_PORT" \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len "$MAX_LENGTH" \
    --disable-log-requests --disable-log-stats
  exec "$PYTHON_BIN" serve_vllm.py \
    --model "$MODEL_NAME" \
    --port "$ROLLOUT_PORT" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len "$MAX_LENGTH" \
    --disable-log-requests \
    --disable-log-stats
fi

: > "$ROLLOUT_LOG"
export CUDA_VISIBLE_DEVICES="$ROLLOUT_CUDA_VISIBLE_DEVICES"
"$PYTHON_BIN" serve_vllm.py \
  --model "$MODEL_NAME" \
  --port "$ROLLOUT_PORT" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len "$MAX_LENGTH" \
  --disable-log-requests \
  --disable-log-stats > "$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!

cleanup() {
  if [[ -n "${ROLLOUT_PID:-}" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
    kill "$ROLLOUT_PID" 2>/dev/null || true
    wait "$ROLLOUT_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM
wait_for_rollout_server "$ROLLOUT_LOG" "$ROLLOUT_TIMEOUT_SECONDS"

CMD=(
  "$ACCELERATE_BIN" launch
  --num_processes 2
  --gpu_ids "$TRAIN_GPU_IDS"
  -m exps_online_rl.codeact_grpo
  --model_name "$MODEL_NAME"
  --data_path "$DATA_PATH"
  --output_root "$OUTPUT_ROOT"
  --seed "$SEED"
  --external_rollout_api_base "$ROLLOUT_API_BASE"
  --num_questions_per_sync "$NUM_QUESTIONS_PER_SYNC"
  --num_rollouts_per_question "$NUM_ROLLOUTS_PER_QUESTION"
  --max_syncs "$MAX_SYNCS"
  --grpo_epochs "$GRPO_EPOCHS"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --kl_lambda "$KL_LAMBDA"
  --save_every_syncs "$SAVE_EVERY_SYNCS"
  --rollout_workers "$ROLLOUT_WORKERS"
  --max_steps "$MAX_STEPS"
  --max_tokens "$MAX_TOKENS"
  --max_length "$MAX_LENGTH"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --python_bin "$PYTHON_BIN"
)

if [[ -n "$RESUME_FROM_ADAPTER" ]]; then
  CMD+=(--resume_from_adapter "$RESUME_FROM_ADAPTER")
fi

printf 'RUN_CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
