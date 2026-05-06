#!/bin/bash
# Step-prefix evaluation pipeline for Qwen3-14B on GSM-hard-300.
# Uses existing scored Qwen3-14B original GSM-300 trajectories.
# Run via:
#   srun --overlap --jobid=7167488 -N1 --ntasks=1 --gpus=4 --mem=0 \
#     bash scripts_modular/run_step_prefix_eval_14b_gsm300.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROOT_DIR
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID="Qwen/Qwen3-14B"
TP_SIZE="4"
PORT="8000"
API_BASE="http://127.0.0.1:${PORT}/v1"

# Existing scored trajectories from Qwen3-14B original GSM-hard-300 run
# (300 total, 300 valid log_data, 199 score==1)
GSM300_SCORED="/scratch/wzhao20/0403test/evaluations/Qwen3-14B_original_gsm_hard_300_20250507_temp=0.7_seed=42_type=agent_steps=5_python_only_python_only_seed42_scored.jsonl"

EVAL_OUT_DIR="/scratch/wzhao20/AKDA2/AgentDistill/logs/step_prefix_eval"
FILTERED_PATH="${EVAL_OUT_DIR}/Qwen3-14B_gsm300_filtered.jsonl"
EVAL_OUT="${EVAL_OUT_DIR}/Qwen3-14B_gsm300_step_prefix_eval_v2.jsonl"

VLLM_PID=""

cleanup_eval() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait  "$VLLM_PID" 2>/dev/null || true
  fi
  cleanup_collection_resources
}
trap cleanup_eval EXIT INT TERM

setup_agentdistill_env   # sets env + cd's to ROOT_DIR
mkdir -p "$EVAL_OUT_DIR"

# ---------------------------------------------------------------------------
# Phase 1: filter — keep only correct, error-free GSM-300 trajectories
# ---------------------------------------------------------------------------
echo "=== Phase 1: Filtering GSM-300 trajectories ==="
"$PYTHON_BIN" - "$GSM300_SCORED" "$FILTERED_PATH" <<'PY'
import json, sys
from pathlib import Path

src  = Path(sys.argv[1])
dest = Path(sys.argv[2])

ERR_PARSE    = "Error in code parsing"
ERR_MAXSTEPS = "Reached max steps"

kept = 0
with src.open() as fin, dest.open('w') as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        entry = json.loads(line)
        if entry.get('score') != 1:
            continue
        messages = (entry.get('log_data') or {}).get('messages', [])
        error = any(
            msg.get('role') == 'tool-response' and
            any(tag in (msg.get('content') or [{}])[0].get('text', '')
                for tag in [ERR_PARSE, ERR_MAXSTEPS])
            for msg in messages
        )
        if error:
            continue
        fout.write(line + '\n')
        kept += 1

print(f'Filtered: {kept} correct error-free trajectories')
PY
echo "Filtered path: ${FILTERED_PATH}"

# ---------------------------------------------------------------------------
# Phase 1b: validate — no None/NaN answers, all scores == 1
# ---------------------------------------------------------------------------
echo "=== Phase 1b: Validating filtered trajectories ==="
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
print("Validation passed.")
PY

# ---------------------------------------------------------------------------
# Phase 2: start vLLM for step-prefix evaluation
# ---------------------------------------------------------------------------
echo "=== Phase 2: Starting vLLM server for step-prefix eval ==="
VLLM_LOG="${EVAL_OUT_DIR}/vllm_gsm300_step_prefix_eval_serve.log"
: > "$VLLM_LOG"

# Use v0 engine to avoid EngineCore subprocess CUDA cgroup issue under SLURM.
export VLLM_USE_V1=0

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
# Phase 3: step-prefix evaluation
# ---------------------------------------------------------------------------
echo "=== Phase 3: Running step-prefix evaluation ==="
"$PYTHON_BIN" -m exps_research.step_prefix_eval \
  --input_jsonl  "$FILTERED_PATH" \
  --model_id     "$MODEL_ID" \
  --api_base     "$API_BASE" \
  --output_jsonl "$EVAL_OUT" \
  --api_key      token-abc \
  --n_samples    5 \
  --temperature  0.7 \
  --top_p        0.8 \
  --max_tokens   1024 \
  --top_logprobs 5

echo "=== Done. Results at: ${EVAL_OUT} ==="
