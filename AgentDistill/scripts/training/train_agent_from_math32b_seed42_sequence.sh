#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"

RAW_LOG="${RAW_LOG:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test/Qwen3-32B_temp=0.7_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl}"
TRAIN_TAG="${TRAIN_TAG:-math32b_seed42_basicdistill}"
EPOCHS="${EPOCHS:-2}"

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
)

if [[ ! -f "$RAW_LOG" ]]; then
  echo "Raw teacher log not found: $RAW_LOG" >&2
  exit 1
fi

python -m exps_research.unified_framework.score_answers \
  --log_files "$RAW_LOG" \
  --task_type math \
  --max_workers 8

SCORED_LOG="$(dirname "$RAW_LOG")/evaluations/$(basename "${RAW_LOG%.jsonl}")_scored.jsonl"
if [[ ! -f "$SCORED_LOG" ]]; then
  echo "Scored log not found: $SCORED_LOG" >&2
  exit 1
fi

python -m exps_research.unified_framework.filter_agent_training_data \
  --result_path "$SCORED_LOG" \
  --do_save

FILTERED_LOG="$(dirname "$RAW_LOG")/filtered_data/$(basename "${RAW_LOG%.jsonl}")_filtered.jsonl"
if [[ ! -f "$FILTERED_LOG" ]]; then
  echo "Filtered log not found: $FILTERED_LOG" >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  echo "=== Training $model from $FILTERED_LOG ==="
  torchrun --nproc_per_node=2 exps_research/finetune_sft.py \
    --model_name "$model" \
    --num_epochs "$EPOCHS" \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr 2e-4 \
    --train_filepath "$FILTERED_LOG" \
    --postfix "$TRAIN_TAG" \
    --solution_type agent \
    --fsdp exps_research/mp_configs/fsdp_2gpu.json \
    --gradient_checkpointing \
    --max_length 10240
done
