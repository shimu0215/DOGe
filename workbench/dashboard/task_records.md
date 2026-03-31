# Task Records

This file is the shared task log for the Hopper dashboard.
Only records submitted in the last 48 hours are shown in the dashboard.
Append new records at the end using the exact structure below.

## Required Format

### TASK-20260324-180500-train-math-student
- submitted_at: 2026-03-24T18:05:00-04:00
- task_name: Train math student LoRA
- job_id: 6610054
- gpu_count: 2
- description: Launches a student LoRA training run for the math seed42 teacher outputs.

### TASK-20260324-191500-eval-math-student
- submitted_at: 2026-03-24T19:15:00-04:00
- task_name: Eval fine-tuned student
- job_id: 6610054
- gpu_count: 1
- description: Runs a gsm-hard evaluation pass on the latest fine-tuned checkpoint.

### TASK-20260324-203000-hold-gpuq-fallback
- submitted_at: 2026-03-24T20:30:00-04:00
- task_name: Hold gpuq fallback reservation
- job_id: 6610053
- gpu_count: 4
- description: Keeps the gpuq reservation alive as the fallback while contrib-gpuq is already running.

### TASK-20260328-205638-train-32b-entropy-retry
- submitted_at: 2026-03-28T20:56:38-04:00
- task_name: Train 32B entropy retry
- job_id: 6629287
- gpu_count: 4
- description: Launches the direct-node Hopper retry of the Qwen3-32B entropy self-finetune script on the active 4xA100 reservation.

### TASK-20260328-223055-train-32b-entropy-deepspeed
- submitted_at: 2026-03-28T22:30:55-04:00
- task_name: Train 32B entropy DeepSpeed
- job_id: 6629287
- gpu_count: 4
- description: Launches the direct-node Hopper retry of the Qwen3-32B entropy self-finetune script using the new TRL plus DeepSpeed path and memory monitoring.

### TASK-20260328-233557-hold-32b-deepspeed-fullnode
- submitted_at: 2026-03-28T23:35:57-04:00
- task_name: Hold 4xA100 full node
- job_id: 6630332
- gpu_count: 4
- description: Keeps a running 4xA100 4-hour contrib-gpuq full-node reservation available for the next 32B DeepSpeed retry.

### TASK-20260330-151946-hold-4xa100-8h-contrib
- submitted_at: 2026-03-30T15:19:46-04:00
- task_name: Hold 4xA100 8h contrib
- job_id: 6637805
- gpu_count: 4
- description: Reserves one 8-hour 4xA100 whole-node slot on contrib-gpuq for follow-up AgentDistill experiments.

### TASK-20260330-175328-hold-4xa100-6h-contrib
- submitted_at: 2026-03-30T17:53:28-04:00
- task_name: Hold 4xA100 6h contrib
- job_id: 6658252
- gpu_count: 4
- description: Reserves one 6-hour 4xA100 whole-node slot on contrib-gpuq for follow-up AgentDistill experiments.
