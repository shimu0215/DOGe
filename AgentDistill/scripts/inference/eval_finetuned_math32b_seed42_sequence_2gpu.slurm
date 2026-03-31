#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --time=08:00:00
#SBATCH --job-name=ad-eval-ft-m32
#SBATCH --output=/scratch/wzhao20/AgentDistill/logs/eval_finetuned_math32b_seed42_%j.log
#SBATCH --error=/scratch/wzhao20/AgentDistill/logs/eval_finetuned_math32b_seed42_%j.log

set -euo pipefail
source /home/wzhao20/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean

cd /scratch/wzhao20/AgentDistill
export HF_HOME=/scratch/wzhao20/hf_cache
export TRANSFORMERS_CACHE=/scratch/wzhao20/hf_cache
export HF_DATASETS_CACHE=/scratch/wzhao20/hf_cache/datasets
export XDG_CACHE_HOME=/scratch/wzhao20/.cache
export VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache
export TRITON_CACHE_DIR=/scratch/wzhao20/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/wzhao20/torchinductor_cache
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

bash scripts/inference/eval_finetuned_math32b_seed42_sequence.sh
