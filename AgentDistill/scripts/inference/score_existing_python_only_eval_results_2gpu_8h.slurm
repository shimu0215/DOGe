#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=ad-score-math-gsm
#SBATCH -o /scratch/wzhao20/AgentDistill/logs/score_existing_python_only_eval_%j.log

set -euo pipefail
source /home/wzhao20/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean
cd /scratch/wzhao20/AgentDistill
bash scripts/inference/score_existing_python_only_eval_results.sh
