#!/bin/bash
# run_baseline_sft_32b.sh — Baseline SFT on original Qwen3-32B teacher trajectories
# Branch: claude/可多拉
#
# This is the control condition: fine-tune teacher using standard SFT (no RL),
# using the same pre-collected trajectories as the RL methods.
# Uses the existing finetune_sft.py pipeline.
#
# Usage:
#   bash scripts/training/run_baseline_sft_32b.sh
#
# Key env overrides:
#   EPOCHS         number of SFT epochs (default 2)
#   LAMBDA         entropy regularisation lambda (default 0, i.e., plain SFT)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- Environment ----
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
export PATH="$CONDA_ENV_PREFIX/bin:${PATH:-}"
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

PYTHON_BIN="$CONDA_ENV_PREFIX/bin/python"
TORCHRUN_BIN="$CONDA_ENV_PREFIX/bin/torchrun"

# ---- Paths ----
RAW_ROOT="${RAW_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/training_outputs/qwen3-32B/baseline_sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-exps_research/mp_configs/ds3_no_offload.json}"

# ---- Hyperparameters ----
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
RAW_MODEL_NAME="${RAW_MODEL_NAME:-Qwen3-32B}"
SEED_START="${SEED_START:-42}"
SEED_END="${SEED_END:-56}"   # inclusive
EPOCHS="${EPOCHS:-2}"
LAMBDA="${LAMBDA:-0.0}"      # 0 = plain SFT; >0 = entropy regularisation
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

mkdir -p logs "$OUTPUT_DIR"
RUN_LOG="$OUTPUT_DIR/run_baseline_sft.log"

echo "=== Baseline SFT 32B ===" | tee "$RUN_LOG"
echo "  model:  $MODEL_NAME"  | tee -a "$RUN_LOG"
echo "  seeds:  $SEED_START..$SEED_END"  | tee -a "$RUN_LOG"
echo "  epochs: $EPOCHS  lambda: $LAMBDA"  | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"

# ---- Collect filtered trajectory paths ----
FILTERED_LOGS=()
for seed in $(seq "$SEED_START" "$SEED_END"); do
    filtered="${RAW_ROOT}/filtered_data/${RAW_MODEL_NAME}_temp=0.7_seed=${seed}_type=agent_steps=5_python_only_python_only_seed${seed}_filtered.jsonl"
    if [[ ! -f "$filtered" ]]; then
        echo "WARNING: filtered file not found: $filtered" | tee -a "$RUN_LOG"
        continue
    fi
    FILTERED_LOGS+=("$filtered")
done

echo "Using ${#FILTERED_LOGS[@]} filtered trajectory files." | tee -a "$RUN_LOG"

if [[ "${#FILTERED_LOGS[@]}" -eq 0 ]]; then
    echo "ERROR: no filtered trajectory files found." | tee -a "$RUN_LOG"
    exit 1
fi

# ---- Preflight check ----
"$PYTHON_BIN" - <<'PY'
import torch, transformers
print(f"torch={torch.__version__}  transformers={transformers.__version__}")
print(f"GPUs: {torch.cuda.device_count()}")
PY

# ---- Build training arguments ----
TRAIN_ARGS=(
    --model_name "$MODEL_NAME"
    --num_epochs "$EPOCHS"
    --batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --lr "$LR"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --max_length "$MAX_LENGTH"
    --train_filepath "${FILTERED_LOGS[@]}"
    --solution_type agent
    --deepspeed "$DEEPSPEED_CONFIG"
    --gradient_checkpointing
    --exp_id "baseline_sft_seeds${SEED_START}_${SEED_END}"
    --postfix "baseline"
)

# Add entropy regularisation if lambda > 0
if [[ "$LAMBDA" != "0" && "$LAMBDA" != "0.0" ]]; then
    TRAIN_ARGS+=(--use_entropy_regularization --entropy_lambda "$LAMBDA")
fi

# ---- Launch ----
"$TORCHRUN_BIN" --nproc_per_node=4 \
    exps_research/finetune_sft.py \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "$RUN_LOG"

echo "=== Baseline SFT complete ===" | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"
