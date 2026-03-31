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
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-5}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
GPU_UTIL="${GPU_UTIL:-0.70}"
LOG_ROOT="${LOG_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_eval}"
DATASETS=(
  "${DATA_PATH_GSM:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/gsm_hard_500_20250507.json}"
  "${DATA_PATH_MATH:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json}"
)

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
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
  local model_name
  local dataset_name
  model_name="$(basename "$model_id")"
  dataset_name="$(basename "$data_path" .json)_test"
  printf "%s/%s/%s_temp=0.0_seed=42_type=agent_steps=5_python_only.jsonl" \
    "$LOG_ROOT" "$dataset_name" "$model_name"
}

scored_summary_for() {
  local result_file="$1"
  local result_dir
  local base_name
  result_dir="$(dirname "$result_file")"
  base_name="$(basename "$result_file" .jsonl)"
  printf "%s/evaluations/evaluation_summary_%s.json" "$result_dir" "$base_name"
}

is_completed() {
  local model_id="$1"
  local data_path="$2"
  local result_file
  result_file="$(output_file_for "$model_id" "$data_path")"

  if [[ -f "$result_file" ]]; then
    if [[ "$(wc -l < "$result_file")" -ge 500 ]]; then
      return 0
    fi
  fi
  return 1
}

run_one_model() {
  local model_id="$1"
  local data_path="$2"
  local model_name
  model_name="$(basename "$model_id")"
  local serve_log="$LOG_ROOT/${model_name}_serve.log"
  local model_tp_size="$TP_SIZE"

  mkdir -p "$LOG_ROOT"
  : > "$serve_log"

  cleanup

  case "$model_name" in
    Qwen3-4B)
      model_tp_size="${TP_SIZE_4B:-1}"
      ;;
    Qwen3-8B)
      model_tp_size="${TP_SIZE_8B:-1}"
      ;;
    *)
      model_tp_size="${TP_SIZE_SMALL:-1}"
      ;;
  esac

  python serve_vllm.py \
    --model "$model_id" \
    --tensor-parallel-size "$model_tp_size" \
    --port "$PORT_BASE" \
    --gpu-memory-utilization "$GPU_UTIL" \
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
    --log_folder "$LOG_ROOT" \
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
    --use_single_endpoint

  cleanup
  VLLM_PID=""
}

trap cleanup EXIT INT TERM

for data_path in "${DATASETS[@]}"; do
  for model_id in "${MODELS[@]}"; do
    if is_completed "$model_id" "$data_path"; then
      echo "=== Skipping completed $model_id on $(basename "$data_path") ==="
      continue
    fi
    echo "=== Evaluating $model_id on $(basename "$data_path") ==="
    run_one_model "$model_id" "$data_path"
  done
done
