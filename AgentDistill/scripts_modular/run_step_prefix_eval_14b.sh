#!/bin/bash
# Step-prefix evaluation pipeline for Qwen3-14B on math_100.
# Uses existing math500 Qwen3-14B scored trajectories; collection already done.
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
TP_SIZE="4"
PORT="8000"
API_BASE="http://127.0.0.1:${PORT}/v1"

# Existing scored trajectories from math500 Qwen3-14B base run
# (contains all 100 math_100 questions, 63 correct)
MATH500_SCORED="/scratch/wzhao20/AKDA2/AgentDistill/logs/qa_results_python_only_teacher/math500_job7074967_20260430_base14b_a1/evaluations/Qwen3-14B_math500_job7074967_20260430_base14b_a1_temp=0.7_seed=42_type=agent_steps=5_python_only_python_only_seed42_scored.jsonl"

EVAL_OUT_DIR="/scratch/wzhao20/AKDA2/AgentDistill/logs/step_prefix_eval"
EVAL_OUT="${EVAL_OUT_DIR}/Qwen3-14B_math_100_step_prefix_eval.jsonl"

# Intermediate paths for the math_100 subset
MATH100_SCORED="${EVAL_OUT_DIR}/Qwen3-14B_math_100_from_math500_scored.jsonl"
FILTERED_PATH="${EVAL_OUT_DIR}/Qwen3-14B_math_100_filtered.jsonl"

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
# Phase 1: extract math_100 subset from existing scored math500 trajectories
# ---------------------------------------------------------------------------
echo "=== Phase 1: Extracting math_100 entries from existing scored math500 data ==="
"$PYTHON_BIN" - "$DATA_PATH" "$MATH500_SCORED" "$MATH100_SCORED" <<'PY'
import json, sys
from pathlib import Path

math100_path = Path(sys.argv[1])
scored_path  = Path(sys.argv[2])
out_path     = Path(sys.argv[3])

with math100_path.open() as f:
    data = json.load(f)
examples = data['examples'] if isinstance(data, dict) and 'examples' in data else data
math100_qs = {e['question'] for e in examples}

written = 0
with scored_path.open() as fin, out_path.open('w') as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        entry = json.loads(line)
        q = entry.get('question') or entry.get('problem') or ''
        if q in math100_qs and isinstance(entry.get('log_data'), dict):
            fout.write(line + '\n')
            written += 1

print(f'Extracted {written} math_100 entries with valid trajectories')
PY
echo "Math_100 scored subset: ${MATH100_SCORED}"

# ---------------------------------------------------------------------------
# Phase 2: filter — keep only correct, error-free trajectories
# ---------------------------------------------------------------------------
echo "=== Phase 2: Filtering trajectories ==="
"$PYTHON_BIN" - "$MATH100_SCORED" "$FILTERED_PATH" <<'PY'
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

if [[ ! -f "$FILTERED_PATH" ]]; then
  echo "ERROR: filtered file not found: ${FILTERED_PATH}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 2b: validate — no None/NaN answers, all scores == 1
# ---------------------------------------------------------------------------
echo "=== Phase 2b: Validating filtered trajectories ==="
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
# Phase 3: start vLLM for step-prefix evaluation
# ---------------------------------------------------------------------------
echo "=== Phase 3: Starting vLLM server for step-prefix eval ==="
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
# Phase 4: step-prefix evaluation
# ---------------------------------------------------------------------------
echo "=== Phase 4: Running step-prefix evaluation ==="
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
