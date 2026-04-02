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

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/scratch/wzhao20/conda_envs/llama-factory311-clean/bin}"
export PATH="$CONDA_ENV_BIN:$PATH"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_BIN/python}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
LORA_FOLDER="${LORA_FOLDER:-/scratch/wzhao20/AKDA2/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda10}"
DATA_PATH="${DATA_PATH:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json}"
SEED="${SEED:-42}"
PORT_BASE="${PORT_BASE:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-5}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
N="${N:-1}"

RESULT_JSONL="${LORA_FOLDER}/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=${N}_seed=${SEED}_type=agent_steps=${MAX_STEPS}_python_only_python_only_seed${SEED}.jsonl"
SERVE_LOG="$(dirname "$RESULT_JSONL")/Qwen3-32B_teacher_collect_seed${SEED}_serve.log"
TMP_DATA_PATH="/tmp/lambda10_remaining_seed${SEED}.json"

pkill -f "run_experiment --experiment_type agent" 2>/dev/null || true
pkill -f "serve_vllm.py --model Qwen/Qwen3-32B" 2>/dev/null || true
pkill -f "vllm serve Qwen/Qwen3-32B" 2>/dev/null || true
sleep 5

"$PYTHON_BIN" - "$DATA_PATH" "$RESULT_JSONL" "$TMP_DATA_PATH" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

with source_path.open() as f:
    source = json.load(f)

examples = source["examples"] if isinstance(source, dict) and "examples" in source else source
completed = set()
if result_path.exists():
    with result_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            q = entry.get("question") or entry.get("problem") or entry.get("prompt")
            if q:
                completed.add(q)

remaining = []
for entry in examples:
    q = entry.get("question") or entry.get("problem") or entry.get("prompt")
    if q in completed:
        continue
    remaining.append(entry)

payload = dict(source) if isinstance(source, dict) and "examples" in source else remaining
if isinstance(source, dict) and "examples" in source:
    payload["examples"] = remaining

with output_path.open("w") as f:
    json.dump(payload, f)

print(f"remaining={len(remaining)}")
PY

"$PYTHON_BIN" serve_vllm.py \
  --model "$MODEL_ID" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT_BASE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --disable-log-requests \
  --disable-log-stats \
  --lora-modules "finetune=$LORA_FOLDER" \
  --max-lora-rank "$MAX_LORA_RANK" > "$SERVE_LOG" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 360); do
  if grep -q "Application startup complete." "$SERVE_LOG" 2>/dev/null; then
    break
  fi
  sleep 5
done

if ! grep -q "Application startup complete." "$SERVE_LOG" 2>/dev/null; then
  echo "Timed out waiting for vLLM startup"
  exit 1
fi

"$PYTHON_BIN" -m exps_research.unified_framework.run_experiment \
  --experiment_type agent \
  --data_path "$TMP_DATA_PATH" \
  --model_type vllm \
  --model_id "$MODEL_ID" \
  --log_folder /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher \
  --max_tokens "$MAX_TOKENS" \
  --multithreading \
  --use_process_pool \
  --parallel_workers "$PARALLEL_WORKERS" \
  --n "$N" \
  --temperature 0.7 \
  --top_p 0.8 \
  --seed "$SEED" \
  --max_steps "$MAX_STEPS" \
  --search_engine_type python_only \
  --use_single_endpoint \
  --task_type math \
  --suffix "python_only_seed${SEED}" \
  --fine_tuned \
  --lora_folder "$LORA_FOLDER"

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
