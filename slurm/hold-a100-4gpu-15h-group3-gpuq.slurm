#!/bin/bash
#SBATCH --job-name=hold-a100-4g-15h-g3
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --time=15:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /scratch/wzhao20/hold_logs/%x-%j.out

sleep 54000
