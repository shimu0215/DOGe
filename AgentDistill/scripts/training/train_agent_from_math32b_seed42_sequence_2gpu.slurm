#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=ad-train-m32-s42
#SBATCH -o /scratch/wzhao20/AgentDistill/logs/train_agent_from_math32b_seed42_%j.log

set -euo pipefail
source /home/wzhao20/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean
cd /scratch/wzhao20/AgentDistill
bash scripts/training/train_agent_from_math32b_seed42_sequence.sh
