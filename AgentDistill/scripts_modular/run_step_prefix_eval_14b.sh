#!/bin/bash
# Step-prefix evaluation pipeline for Qwen3-14B on math_100.
# Run via:
#   srun --overlap --jobid=7166025 -N1 --ntasks=1 --gpus=4 --mem=0 \
#     bash scripts_modular/run_step_prefix_eval_14b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROOT_DIR
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID="Qwen/Qwen3-14B"
DATA_PATH="/scratch/wzhao20/0403test/math_100_20250414.json"
SEED="42"
TP_SIZE="4"
TEMPERATURE="0.7"
MAX_STEPS="5"
N="1"
LOG_ROOT="/scratch/wzhao20/AKDA2/AgentDistill/logs/qa_results_python_only_teacher"
OUTPUT_NAME_TAG="math_100_20250414"
SUFFIX_TAG="python_only_seed${SEED}"
PORT="8000"
API_BASE="http://127.0.0.1:${PORT}/v1"

EVAL_OUT_DIR="/scratch/wzhao20/AKDA2/AgentDistill/logs/step_prefix_eval"
EVAL_OUT="${EVAL_OUT_DIR}/Qwen3-14B_math_100_step_prefix_eval.jsonl"

VLLM_PID=""

cleanup_eval() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait  "$VLLM_PID" 2>/dev/null || true
  fi
  cleanup_collection_resources
}
trap cleanup_eval EXIT INT TERM

# ---------------------------------------------------------------------------
# Phase 1: collect trajectories with Qwen3-14B
# ---------------------------------------------------------------------------
echo "=== Phase 1: Collecting math_100 trajectories ==="
bash "${SCRIPT_DIR}/collect_unit.sh" \
  --model-id        "$MODEL_ID" \
  --data-path       "$DATA_PATH" \
  --seed            "$SEED" \
  --tp-size         "$TP_SIZE" \
  --temperature     "$TEMPERATURE" \
  --max-steps       "$MAX_STEPS" \
  --n               "$N" \
  --log-root        "$LOG_ROOT" \
  --output-name-tag "$OUTPUT_NAME_TAG"

# collect_unit.sh kills vLLM on exit, so we can safely restart it later.

setup_agentdistill_env   # sets env + cd's to ROOT_DIR

RAW_RESULT="$(result_jsonl_path \
  "$MODEL_ID" "$DATA_PATH" "$SEED" "$MAX_STEPS" "$N" \
  "" "$LOG_ROOT" "$OUTPUT_NAME_TAG" "$TEMPERATURE" "$SUFFIX_TAG")"
echo "Raw result: ${RAW_RESULT}"

# ---------------------------------------------------------------------------
# Phase 2: score answers (math grader, no external model)
# ---------------------------------------------------------------------------
echo "=== Phase 2: Scoring answers ==="
"$PYTHON_BIN" -m exps_research.unified_framework.score_answers \
  --log_files "$RAW_RESULT" \
  --task_type math \
  --single_thread

RAW_DIR="$(dirname "$RAW_RESULT")"
RAW_BASE="$(basename "$RAW_RESULT" .jsonl)"
SCORED_PATH="${RAW_DIR}/evaluations/${RAW_BASE}_scored.jsonl"
echo "Scored path: ${SCORED_PATH}"

# ---------------------------------------------------------------------------
# Phase 3: filter — keep only correct, error-free trajectories
# ---------------------------------------------------------------------------
echo "=== Phase 3: Filtering trajectories ==="
"$PYTHON_BIN" -m exps_research.unified_framework.filter_agent_training_data \
  --result_path "$SCORED_PATH" \
  --do_save

FILTERED_PATH="${SCORED_PATH/\/evaluations\//\/filtered_data\/}"
FILTERED_PATH="${FILTERED_PATH/_scored.jsonl/_filtered.jsonl}"
echo "Filtered path: ${FILTERED_PATH}"

if [[ ! -f "$FILTERED_PATH" ]]; then
  echo "ERROR: filtered file not found: ${FILTERED_PATH}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 3b: validate — no None/NaN answers, all scores == 1
# ---------------------------------------------------------------------------
echo "=== Phase 3b: Validating filtered trajectories ==="
"$PYTHON_BIN" - "$FILTERED_PATH" <<'PY'
import json, sys

path = sys.argv[1]
n_invalid = 0
with open(path) as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        ans = str(entry.get("true_answer") or entry.get("answer") or "").strip()
        score = entry.get("score")
        if not ans or ans.lower() in ("none", "nan"):
            print(f"  [line {i+1}] INVALID answer: {ans!r}  q={str(entry.get('question',''))[:60]!r}")
            n_invalid += 1
        if score != 1:
            print(f"  [line {i+1}] Non-correct entry (score={score}), should have been filtered")
            n_invalid += 1

if n_invalid > 0:
    print(f"ERROR: {n_invalid} invalid entries. Aborting.")
    sys.exit(1)
print(f"Validation passed.")
PY

# ---------------------------------------------------------------------------
# Phase 4: start vLLM for step-prefix evaluation
# ---------------------------------------------------------------------------
echo "=== Phase 4: Starting vLLM server for step-prefix eval ==="
mkdir -p "$EVAL_OUT_DIR"
VLLM_LOG="${EVAL_OUT_DIR}/vllm_step_prefix_eval_serve.log"
: > "$VLLM_LOG"

"$PYTHON_BIN" serve_vllm.py \
  --model                    "$MODEL_ID" \
  --tensor-parallel-size     "$TP_SIZE" \
  --port                     "$PORT" \
  --gpu-memory-utilization   0.85 \
  --disable-log-requests \
  --disable-log-stats \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

wait_for_server "$VLLM_LOG" 1800 "$VLLM_PID"
echo "vLLM server ready (pid=${VLLM_PID})."

# ---------------------------------------------------------------------------
# Phase 5: step-prefix evaluation
# ---------------------------------------------------------------------------
echo "=== Phase 5: Running step-prefix evaluation ==="
"$PYTHON_BIN" -m exps_research.step_prefix_eval \
  --input_jsonl  "$FILTERED_PATH" \
  --model_id     "$MODEL_ID" \
  --api_base     "$API_BASE" \
  --output_jsonl "$EVAL_OUT" \
  --n_samples    5 \
  --temperature  0.7 \
  --top_p        0.8 \
  --max_tokens   1024 \
  --top_logprobs 5

echo "=== Done. Results at: ${EVAL_OUT} ==="
