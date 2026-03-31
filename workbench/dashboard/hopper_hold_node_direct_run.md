# Hopper Hold + Node-Direct Run Method

This note records the method that actually worked when `sleep hold + srun --jobid ...` did not.

## Goal

Keep using a `sleep` hold job to reserve GPU resources, but bypass the broken `srun` step-launch path and run the real task directly on the allocated compute node.

## When To Use This

Use this method when:
- the `sleep` hold job itself starts correctly,
- but even the smallest `srun --jobid <jobid> ...` command fails with:
  - `StepId=... aborted before step completely launched`
- and we still need to make progress on the actual workload.

## Core Idea

1. Submit a `sleep` hold Slurm job normally.
2. Wait until the hold is `RUNNING` and note the allocated node.
3. SSH to Hopper, then SSH to the allocated node.
4. Run the real workload directly on that node.
5. Keep all runtime paths inside `/scratch/wzhao20`.

This preserves the resource reservation while bypassing the broken `srun` step-launch layer.

## Example Hold

Example 1xA100 hold:

```bash
#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=02:00:00
#SBATCH --job-name=iter-a100-1h-gq
#SBATCH --output=hpc-results/iter-a100-1h-gq-%j.out
#SBATCH --error=hpc-results/iter-a100-1h-gq-%j.err

set -euo pipefail
mkdir -p hpc-results
hostname
nvidia-smi
sleep 7200
```

## Step 1: Confirm The Hold And Node

```bash
ssh hopper 'cd /scratch/wzhao20 && squeue -j <jobid> -o "%.18i %.9P %.20j %.2t %.10M %R"'
```

Example result:
- job id: `6610039`
- node: `gpu009`

## Step 2: SSH To The Allocated Node

```bash
ssh hopper 'cd /scratch/wzhao20/AgentDistill && ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null gpu009'
```

## Step 3: Activate The Correct Runtime Environment

```bash
cd /scratch/wzhao20/AgentDistill
source /home/wzhao20/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/wzhao20/conda_envs/llama-factory311-clean
```

## Step 4: Force AgentDistill To Use Its Own `smolagents`

This was required to avoid importing the wrong site-packages version of `smolagents`.

```bash
export PYTHONPATH=/scratch/wzhao20/AgentDistill/src:${PYTHONPATH:-}
```

Without this, we hit:

```text
ImportError: cannot import name 'VLLMServerModel' from 'smolagents'
```

## Step 5: Export Cache Paths

```bash
export HF_HOME=/scratch/wzhao20/hf_cache
export TRANSFORMERS_CACHE=/scratch/wzhao20/hf_cache
export HF_DATASETS_CACHE=/scratch/wzhao20/hf_cache/datasets
export XDG_CACHE_HOME=/scratch/wzhao20/.cache
export VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache
export TRITON_CACHE_DIR=/scratch/wzhao20/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/wzhao20/torchinductor_cache
```

## Step 6: Smoke Test Command Chain

This exact smoke test was used to prove the end-to-end AgentDistill flow works outside the broken `srun` path.

### Prepare a tiny dataset

```bash
python - <<'PY'
import json
src="/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/aime_90_20250504.json"
dst="/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/aime_3_smoketest.json"
with open(src) as f:
    data=json.load(f)
data["examples"] = data["examples"][:3]
with open(dst, "w") as f:
    json.dump(data, f)
print(dst)
PY
```

### Clean output directory

```bash
rm -rf /scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b
mkdir -p /scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b
```

### Start vLLM

```bash
python serve_vllm.py \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --port 8018 \
  --gpu-memory-utilization 0.6 \
  --disable-log-requests \
  --disable-log-stats \
  > /scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b/serve.log 2>&1 &
VLLM_PID=$!
```

### Wait for readiness

```bash
for i in $(seq 1 180); do
  grep -q "Application startup complete." /scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b/serve.log && break
  sleep 2
done
```

### Run AgentDistill native generation + evaluation

```bash
python -m exps_research.unified_framework.run_experiment \
  --experiment_type agent \
  --data_path /scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/aime_3_smoketest.json \
  --model_type vllm \
  --model_id Qwen/Qwen3-0.6B \
  --log_folder /scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b \
  --max_tokens 512 \
  --multithreading \
  --use_process_pool \
  --parallel_workers 1 \
  --n 1 \
  --temperature 0.7 \
  --top_p 0.8 \
  --seed 42 \
  --max_steps 5 \
  --search_engine_type python_only \
  --use_single_endpoint \
  --suffix smoke
```

### Stop vLLM

```bash
kill $VLLM_PID || true
wait $VLLM_PID || true
```

## Smoke Test Outputs Observed

The following files were produced successfully:

- `/scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b/aime_3_smoketest_test/Qwen3-0.6B_temp=0.7_seed=42_type=agent_steps=5_python_only_smoke.jsonl`
- `/scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b/aime_3_smoketest_test/evaluations/Qwen3-0.6B_temp=0.7_seed=42_type=agent_steps=5_python_only_smoke_scored.jsonl`
- `/scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b/aime_3_smoketest_test/evaluations/evaluation_summary_Qwen3-0.6B_temp=0.7_seed=42_type=agent_steps=5_python_only_smoke.json`
- `/scratch/wzhao20/AgentDistill/logs/smoke_test_qwen3_0p6b/serve.log`

Observed summary:
- processed 3 questions
- 3/3 correct
- 100%

## Main Conclusion

At the time of writing:
- the AgentDistill native command chain works,
- conda activation works,
- `serve_vllm.py` works,
- `run_experiment` works,
- native evaluation works,
- but `sleep hold + srun --jobid ...` was failing earlier than the application layer.

So this node-direct method is the known-good fallback whenever the hold reservation exists but `srun` step launch is unhealthy.
