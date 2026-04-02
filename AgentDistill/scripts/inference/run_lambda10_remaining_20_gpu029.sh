#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/scratch/wzhao20/conda_envs/llama-factory311-clean/bin}"
export PATH="$CONDA_ENV_BIN:$PATH"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_BIN/python}"

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

MODEL_ID="Qwen/Qwen3-32B"
LORA_FOLDER="/scratch/wzhao20/AKDA2/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda10"
DATA_PATH="/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json"
RESULT_JSONL="${LORA_FOLDER}/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
TMP_DATA_PATH="/tmp/lambda10_remaining_seed42.json"
RUN_LOG="/scratch/wzhao20/AKDA2/AgentDistill/logs/run_experiment_lambda10_remaining_gpu029.log"
TMP_QA_DIR="${LORA_FOLDER}/qa_results/lambda10_remaining_seed42_tmp"
TMP_RESULT_JSONL="${TMP_QA_DIR}/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"

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

remaining = [e for e in examples if (e.get("question") or e.get("problem") or e.get("prompt")) not in completed]
payload = dict(source) if isinstance(source, dict) and "examples" in source else remaining
if isinstance(source, dict) and "examples" in source:
    payload["examples"] = remaining

with output_path.open("w") as f:
    json.dump(payload, f)

print(f"remaining={len(remaining)}")
PY

rm -rf "$TMP_QA_DIR"
: > "$RUN_LOG"

"$PYTHON_BIN" -m exps_research.unified_framework.run_experiment \
  --experiment_type agent \
  --data_path "$TMP_DATA_PATH" \
  --model_type vllm \
  --model_id "$MODEL_ID" \
  --log_folder /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher \
  --max_tokens 1024 \
  --multithreading \
  --use_process_pool \
  --parallel_workers 4 \
  --n 1 \
  --temperature 0.7 \
  --top_p 0.8 \
  --seed 42 \
  --max_steps 5 \
  --search_engine_type python_only \
  --use_single_endpoint \
  --task_type math \
  --suffix python_only_seed42 \
  --fine_tuned \
  --lora_folder "$LORA_FOLDER" \
  >> "$RUN_LOG" 2>&1

"$PYTHON_BIN" - "$RESULT_JSONL" "$TMP_RESULT_JSONL" <<'PY'
import json
import sys
from pathlib import Path

main_path = Path(sys.argv[1])
tmp_path = Path(sys.argv[2])
if not tmp_path.exists():
    raise SystemExit

seen = set()
rows = []
if main_path.exists():
    with main_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line)
            try:
                entry = json.loads(line)
            except Exception:
                continue
            q = entry.get("question") or entry.get("problem") or entry.get("prompt")
            if q:
                seen.add(q)

with tmp_path.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        q = entry.get("question") or entry.get("problem") or entry.get("prompt")
        if q and q in seen:
            continue
        if q:
            seen.add(q)
        rows.append(line)

main_path.parent.mkdir(parents=True, exist_ok=True)
with main_path.open("w") as f:
    for line in rows:
        f.write(line + "\n")
PY
