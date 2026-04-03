#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
export PATH="$CONDA_ENV_PREFIX/bin:${PATH:-}"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

cleanup_compute() {
  pkill -f "run_experiment --experiment_type agent" 2>/dev/null || true
  pkill -f "torchrun" 2>/dev/null || true
  pkill -f "finetune_sft.py" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "serve_vllm.py" 2>/dev/null || true
  sleep 10
}

run_one_teacher() {
  local raw_log="$1"
  local train_tag="$2"

  cleanup_compute

  RAW_LOG="$raw_log" \
  STUDENT_MODEL="Qwen/Qwen3-0.6B" \
  TRAIN_TAG="$train_tag" \
  EPOCHS="${EPOCHS:-2}" \
  DATASET_SIZE="${DATASET_SIZE:--1}" \
  bash scripts/training/train_student_from_single_teacher_math_python.sh

  cleanup_compute
}

collect_ft_teacher() {
  local lora_folder="$1"

  cleanup_compute

  MODEL_ID="Qwen/Qwen3-32B" \
  LORA_FOLDER="$lora_folder" \
  SEED="42" \
  N="1" \
  FORCE_RERUN="${FORCE_RERUN:-0}" \
  bash scripts/inference/collect_teacher_math_python_singletraj.sh

  cleanup_compute
}

for lambda in 2 5 10; do
  lambda_tag="${lambda/./p}"
  teacher_dir="/scratch/wzhao20/AKDA2/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda${lambda_tag}"
  raw_log="$teacher_dir/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"

  collect_ft_teacher "$teacher_dir"
  run_one_teacher "$raw_log" "math32b_lambda${lambda_tag}_singletraj_basicdistill"
done

cleanup_compute
