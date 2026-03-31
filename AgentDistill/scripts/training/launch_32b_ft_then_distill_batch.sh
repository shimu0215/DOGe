#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PATH="/scratch/wzhao20/conda_envs/llama-factory311-clean/bin:${PATH}"

bash scripts/training/train_teacher_entropy_math_32b_owntraj_deepspeed.sh

EPOCHS="${EPOCHS:-2}" \
DATASET_SIZE="${DATASET_SIZE:--1}" \
FORCE_RERUN="${FORCE_RERUN:-1}" \
bash scripts/training/run_qwen3_0p6b_from_32b_teachers_math_python.sh
