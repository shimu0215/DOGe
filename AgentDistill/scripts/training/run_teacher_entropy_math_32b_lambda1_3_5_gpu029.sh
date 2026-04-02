#!/bin/bash
set -euo pipefail

cd /scratch/wzhao20/AKDA2/AgentDistill

export CONDA_ENV_PREFIX=/scratch/wzhao20/conda_envs/llama-factory311-clean
export PYTHONPATH=/scratch/wzhao20/AKDA2/AgentDistill/src
export PYTHON_BIN=/scratch/wzhao20/conda_envs/llama-factory311-clean/bin/python
export TORCHRUN_BIN=/scratch/wzhao20/conda_envs/llama-factory311-clean/bin/torchrun
export LAMBDAS="1 3 5"

exec bash scripts/training/train_teacher_entropy_math_32b_owntraj_deepspeed.sh
