#!/bin/bash
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

PORT_BASE="${PORT_BASE:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-5}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
GPU_UTIL="${GPU_UTIL:-0.70}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

DATASETS=(
  "/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/gsm_hard_500_20250507.json"
  "/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json"
)

BASE_MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
)

LORA_FOLDERS=(
  "/scratch/wzhao20/AgentDistill/training_outputs/qwen-0.5B-instruct/agent_baseline_2epochs_math32b_seed42_basicdistill"
  "/scratch/wzhao20/AgentDistill/training_outputs/qwen-1.5B-instruct/agent_baseline_2epochs_math32b_seed42_basicdistill"
  "/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math32b_seed42_basicdistill"
  "/scratch/wzhao20/AgentDistill/training_outputs/qwen3-1.7B/agent_baseline_2epochs_math32b_seed42_basicdistill"
  "/scratch/wzhao20/AgentDistill/training_outputs/qwen3-4B/agent_baseline_2epochs_math32b_seed42_basicdistill"
  "/scratch/wzhao20/AgentDistill/training_outputs/qwen3-8B/agent_baseline_2epochs_math32b_seed42_basicdistill"
)

VLLM_PID=""

cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "serve_vllm.py" 2>/dev/null || true
  sleep 3
}

wait_for_server() {
  local log_file="$1"
  local timeout_s="${2:-900}"
  local waited=0
  until grep -q "Application startup complete." "$log_file" 2>/dev/null; do
    if (( waited >= timeout_s )); then
      echo "Timed out waiting for server startup: $log_file" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

output_file_for() {
  local model_id="$1"
  local data_path="$2"
  local lora_folder="$3"
  local model_name
  local dataset_name
  model_name="$(basename "$model_id")"
  dataset_name="$(basename "$data_path" .json)_test"
  printf "%s/qa_results/%s/%s_temp=0.0_seed=42_type=agent_steps=5_python_only.jsonl" \
    "$lora_folder" "$dataset_name" "$model_name"
}

is_completed() {
  local model_id="$1"
  local data_path="$2"
  local lora_folder="$3"
  local result_file
  result_file="$(output_file_for "$model_id" "$data_path" "$lora_folder")"
  if [[ -f "$result_file" ]] && [[ "$(wc -l < "$result_file")" -ge 500 ]]; then
    return 0
  fi
  return 1
}

score_one_result() {
  local result_file="$1"
  python -m exps_research.unified_framework.score_answers \
    --log_files "$result_file" \
    --task_type math \
    --max_workers 8
}

run_one_model() {
  local model_id="$1"
  local data_path="$2"
  local lora_folder="$3"
  local model_name serve_log result_file
  model_name="$(basename "$model_id")"
  serve_log="$lora_folder/qa_results/${model_name}_serve.log"
  result_file="$(output_file_for "$model_id" "$data_path" "$lora_folder")"

  mkdir -p "$(dirname "$result_file")"
  : > "$serve_log"

  cleanup

  python serve_vllm.py \
    --model "$model_id" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$PORT_BASE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --lora-modules finetune="$lora_folder" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --disable-log-requests \
    --disable-log-stats \
    > "$serve_log" 2>&1 &
  VLLM_PID=$!

  wait_for_server "$serve_log"

  python -m exps_research.unified_framework.run_experiment \
    --experiment_type agent \
    --data_path "$data_path" \
    --model_type vllm \
    --model_id "$model_id" \
    --max_tokens "$MAX_TOKENS" \
    --multithreading \
    --use_process_pool \
    --parallel_workers "$PARALLEL_WORKERS" \
    --n 1 \
    --temperature 0.0 \
    --top_p 0.8 \
    --seed 42 \
    --max_steps "$MAX_STEPS" \
    --search_engine_type python_only \
    --use_single_endpoint \
    --fine_tuned \
    --lora_folder "$lora_folder"

  if [[ -f "$result_file" ]]; then
    score_one_result "$result_file"
  fi

  cleanup
  VLLM_PID=""
}

trap cleanup EXIT INT TERM

for data_path in "${DATASETS[@]}"; do
  for i in "${!BASE_MODELS[@]}"; do
    model_id="${BASE_MODELS[$i]}"
    lora_folder="${LORA_FOLDERS[$i]}"
    if is_completed "$model_id" "$data_path" "$lora_folder"; then
      echo "=== Skipping completed $model_id on $(basename "$data_path") ==="
      continue
    fi
    echo "=== Evaluating finetuned $model_id on $(basename "$data_path") ==="
    run_one_model "$model_id" "$data_path" "$lora_folder"
  done
done
