#!/bin/bash
#SBATCH --job-name=akda-1022
#SBATCH --qos=gpu  # or gpu/cs_dept
#SBATCH --partition=contrib-gpuq   # or gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --output=akda-1022-out.out   # Output file
#SBATCH --error=akda-1022-error.err    # Error file
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=2:00:00

# keep whatever you want
source ~/.bashrc
cd /scratch/wzhao20/agent-distillation/agent-distillation
module load cuda/12.6
conda activate agents



# nsys profile --delay=120 --duration=300 --output=dp16 --trace=cuda,osrt \
#  --sampling-period=28000000 --force-overwrite=true --stats=true \
#  ./examples/llama/train_llama3_1b_a100_dp16.sh
# python xxx.py
# srun xxx.py
bash /scratch/wzhao20/agent-distillation/agent-distillation/scripts/inference/run_agent_teacher_train.sh