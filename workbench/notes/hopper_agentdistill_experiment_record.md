# Hopper AgentDistill Experiment Record

Last updated: 2026-03-25 23:35 America/New_York
Recorded by: Codex
Workspace root: `/Users/shimu/Downloads/DOGe-main`
Primary remote root: `/scratch/wzhao20`

## 1. Scope
This record is the handoff document for the ongoing AgentDistill-based experiments on Hopper. It is intended to be sufficient for continuing the work in a fresh conversation using only:
- this document
- the code in this repo
- the remote files under `/scratch/wzhao20`

It focuses on:
- teacher data collection
- native AgentDistill evaluation
- student distillation
- entropy-regularized teacher self-finetuning
- Hopper reservation and launch workflow
- known failure modes and what was learned

## 2. Repos and directories

### Local repos
- AgentDistill: `/Users/shimu/Downloads/DOGe-main/AgentDistill`
- LlamaFactory: `/Users/shimu/Downloads/DOGe-main/LlamaFactory-main`
- Shared notes/dashboard: `/Users/shimu/Downloads/DOGe-main/workbench`

### Important local note files
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/current_hopper_status.md`
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/hopper_hold_node_direct_run.md`
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/teacher_entropy_experiment.md`
- This file: `/Users/shimu/Downloads/DOGe-main/workbench/notes/hopper_agentdistill_experiment_record.md`

### Remote repos
- AgentDistill: `/scratch/wzhao20/AgentDistill`
- LlamaFactory: `/scratch/wzhao20/llama_factory`

## 3. Ground rules used in this project
- Prefer AgentDistill native codepaths for generation, scoring, filtering, training, and eval.
- Restrict Hopper work to `/scratch/wzhao20`.
- Follow `local edit -> local check -> git commit/push -> Hopper git pull --ff-only -> run`.
- Use `python_only` tools for AgentDistill agent runs.
- Do not use prefix memory / CoT memory / planning unless explicitly requested.
- For math-like tasks, scoring should not require OpenAI API keys.

## 4. What was successfully completed

### 4.1 Native AgentDistill eval for base models
These evals were completed using AgentDistill native evaluation with `python_only`, `temperature=0.0`, `n=1`, `max_steps=5`, `max_tokens=1024`.

Datasets:
- GSM-hard: `/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/gsm_hard_500_20250507.json`
- MATH: `/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json`

Models evaluated:
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`

Scoring summary file:
- `/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_eval/scoring_summaries/agent_python_only_eval_summary.json`

Known summary values from that file:
- GSM-hard:
  - 0.5B: 5.6%
  - 1.5B: 28.4%
  - 0.6B: 29.4%
  - 1.7B: 61.4%
  - 4B: 70.8%
  - 8B: 76.6%
- MATH:
  - 0.5B: 4.2%
  - 1.5B: 28.0%
  - 0.6B: 17.6%
  - 1.7B: 46.69% (499 examples in summary)
  - 4B: 58.92% (499 examples in summary)
  - 8B: 63.2%

### 4.2 Teacher trajectory collection already finished
Using AgentDistill native data collection with `python_only`, `temperature=0.7`, 15 seeds (`42..56`):

Completed:
- `Qwen/Qwen3-32B` on GSM-hard: complete
- `Qwen/Qwen3-14B` on GSM-hard: complete
- `Qwen/Qwen3-32B` on MATH: complete
- `Qwen/Qwen3-14B` on MATH: complete
- `Qwen/Qwen3-32B` on AIME: complete
- `Qwen/Qwen3-14B` on AIME: complete

At last known check, still in progress:
- `Qwen/Qwen3-32B` on OlymMATH:
  - seeds `42..51` complete at `200/200`
  - `seed52` at `98/200`
- `Qwen/Qwen3-14B` on OlymMATH`: not started yet

Important remote result roots:
- `/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/gsm_hard_500_20250507_test`
- `/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test`
- `/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/aime_90_20250504_test`
- `/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/olymath_200_20250511_test`

### 4.3 Student distillation already completed
Native AgentDistill basic agent distillation from `Qwen3-32B / MATH / seed42` was completed for six students.

Outputs exist under `/scratch/wzhao20/AgentDistill/training_outputs/`:
- `qwen-0.5B-instruct/agent_baseline_2epochs_math32b_seed42_basicdistill`
- `qwen-1.5B-instruct/agent_baseline_2epochs_math32b_seed42_basicdistill`
- `qwen3-0.6B/agent_baseline_2epochs_math32b_seed42_basicdistill`
- `qwen3-1.7B/agent_baseline_2epochs_math32b_seed42_basicdistill`
- `qwen3-4B/agent_baseline_2epochs_math32b_seed42_basicdistill`
- `qwen3-8B/agent_baseline_2epochs_math32b_seed42_basicdistill`

These are standard LoRA SFT outputs.

## 5. Important AgentDistill code modifications already made
These modifications are local and were pushed during the project. They are intentional and should be preserved unless replaced by something better.

### 5.1 Math scoring should not require API keys
File:
- `/Users/shimu/Downloads/DOGe-main/AgentDistill/exps_research/unified_framework/score_answers.py`

Change:
- Math tasks (`task_type=math`) do local scoring and should not require `keys/openai-key/key.env`.

### 5.2 Native `python_only` generation/eval sequence scripts
Key scripts:
- `/Users/shimu/Downloads/DOGe-main/AgentDistill/scripts/inference/run_agent_python_only_data_sequence.sh`
- `/Users/shimu/Downloads/DOGe-main/AgentDistill/scripts/inference/run_agent_python_only_eval_sequence.sh`

Key behavior:
- Use AgentDistill native `run_experiment`
- `python_only`
- dataset lists and model lists can be overridden
- expected counts come from actual dataset size, not hard-coded 500
- supports AIME (`90`) and OlymMATH (`200`)
- skip/retry logic added for seed-level collection

### 5.3 Entropy self-finetune support added
Files:
- `/Users/shimu/Downloads/DOGe-main/AgentDistill/exps_research/finetune_sft.py`
- `/Users/shimu/Downloads/DOGe-main/AgentDistill/exps_research/train_utils/preprocess.py`

Implemented features:
- `RandomTrajectoryDataset`
  - randomly samples one trajectory per question from grouped trajectory sets
- `EntropyRegularizedSFTTrainer`
  - total loss is standard SFT CE minus entropy regularization on labeled response tokens
- grouped preprocessing from multiple filtered trajectory files
- CLI flags:
  - `--random_trajectory_per_question`
  - `--use_entropy_regularization`
  - `--entropy_lambda`
  - later added:
    - `--lora_r`
    - `--lora_alpha`

Mathematical idea implemented:
- Randomly choose one teacher trajectory for each question as the supervised sequence.
- Use standard CE next-token supervision on that sampled trajectory.
- Add a negative-entropy term on the model output distribution at labeled response positions.
- Current implemented form is essentially:
  - `L = L_sft - lambda * mean_token_entropy`

## 6. Entropy self-finetune experiment history

### 6.1 Goal
Test whether increasing teacher output entropy can make distillation harder.

### 6.2 First sequence experiment
Script:
- `/Users/shimu/Downloads/DOGe-main/AgentDistill/scripts/training/train_teacher_entropy_math_sequence.sh`

Behavior:
- scores and filters `Qwen3-32B / MATH / seed42..56`
- attempts self-finetune with entropy regularization
- multiple lambdas (`0.2`, `0.5`, `1.0`)
- started from `Qwen/Qwen3-32B`
- in practice, 32B did not finish and the sequence fell back to 14B

### 6.3 What successfully trained
`Qwen3-14B` entropy self-finetunes succeeded for:
- `lambda=0.2`
- `lambda=0.5`
- `lambda=1.0`

Output dirs:
- `/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math32b_entropy_self_lambda0p2`
- `/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math32b_entropy_self_lambda0p5`
- `/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math32b_entropy_self_lambda1p0`

These contain full LoRA artifacts.

### 6.4 32B-only retry script
Script:
- `/Users/shimu/Downloads/DOGe-main/AgentDistill/scripts/training/train_teacher_entropy_math_32b_only.sh`

Purpose:
- retry only `Qwen3-32B`
- use dedicated logs
- isolate the failure from the multi-model/multi-lambda sequence

Important commits in the retry sequence:
- `fda34e2` add dedicated 32b entropy retry script
- `26a5823` reduce 32b entropy retry memory pressure
- `f60e9ac` further reduce 32b entropy training footprint
- `c9441f4` make 32b entropy retry more conservative

What those changes did over time:
1. initial 32B-only retry:
   - `max_length=8192`
   - `batch_size=1`
   - `grad_accum=2`
   - default LoRA `r=64, alpha=128`
2. first reduction:
   - `max_length=4096`
   - `grad_accum=4`
3. second reduction:
   - `max_length=2048`
   - `lora_r=16`
   - `lora_alpha=32`
4. third reduction:
   - `max_length=1024`
   - `lora_r=8`
   - `lora_alpha=16`
   - removed `--gradient_checkpointing`

### 6.5 32B conclusion so far
Repeated 32B attempts on 4xA100 80GB using the current AgentDistill/HF/TRL/FSDP path failed in the same way:
- training reaches `torchrun`
- loads model checkpoint shards
- dies during/just after full checkpoint shard loading
- root cause is always a rank receiving `SIGKILL` (`exitcode -9`)

Observed patterns:
- happens after `Loading checkpoint shards: 17/17` or later during shard loading
- happens even after reducing sequence length and LoRA rank significantly
- happens on the correct GPU node (e.g. `gpu024.orc.gmu.edu`)

Interpretation:
- this is likely a resource peak / system kill problem in the current FSDP path
- not a logic error in entropy loss itself
- not a front-end launcher problem
- not a data preprocessing problem

Working conclusion:
- `Qwen3-32B` entropy self-finetune is not stable in the current 4xA100 + AgentDistill FSDP setup
- `Qwen3-14B` entropy self-finetune is the currently viable route for the science question

## 7. Hopper interaction lessons

### 7.1 Most important lesson: `hold + srun --jobid` is unreliable here
The original desired pattern was:
1. submit a `sleep` hold job
2. wait for `RUNNING`
3. run real work with `srun --jobid <hold> --overlap ...`

However, in this project, even minimal commands often failed with:
- `srun: StepId=... aborted before step completely launched`

This was reproduced on fresh holds, so it was not just a stale step issue.

### 7.2 What actually worked reliably
Use the hold only to reserve the compute node, then:
1. identify the allocated node
2. SSH directly to that node
3. run the task there directly

This is the "hold + direct-node-run" pattern.

Documented in:
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/hopper_hold_node_direct_run.md`

### 7.3 Critical environment settings for direct-node runs
Always set:
```bash
export PYTHONPATH=/scratch/wzhao20/AgentDistill/src:${PYTHONPATH:-}
export HF_HOME=/scratch/wzhao20/hf_cache
export TRANSFORMERS_CACHE=/scratch/wzhao20/hf_cache
export HF_DATASETS_CACHE=/scratch/wzhao20/hf_cache/datasets
export XDG_CACHE_HOME=/scratch/wzhao20/.cache
export TRITON_CACHE_DIR=/scratch/wzhao20/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/wzhao20/torchinductor_cache
```

The `PYTHONPATH` fix is important so that AgentDistill uses its own `src/smolagents` rather than an incompatible environment-installed package.

### 7.4 Hopper SSH caveats
Observed issues:
- Duo autopush sometimes triggers anomalous request rate limits
- SSH multiplexing sometimes caused:
  - `mux_client_request_session: session request failed: Session open refused by peer`
- sometimes DNS/network from local machine temporarily failed

Useful troubleshooting:
- use a direct, minimal SSH command first
- if multiplex causes trouble, disable it temporarily
- if Duo is rate-limited, wait a minute before retrying

## 8. Reservation / slurm script locations

Most recent local hold scripts are under:
- `/Users/shimu/Downloads/DOGe-main/LlamaFactory-main/slurm/trl`

Examples:
- `hold-a100-1gpu-iter-2h-gpuq.slurm`
- `hold-a100-1gpu-iter-2h-contrib.slurm`
- `hold-a100-2gpu-iter-8h-gpuq.slurm`
- `hold-a100-2gpu-iter-8h-contrib.slurm`
- `hold-a100-4gpu-iter-8h-gpuq.slurm`
- `hold-a100-4gpu-iter-8h-contrib.slurm`

Remote copies usually live under:
- `/scratch/wzhao20/llama_factory/slurm/trl`

Hold scripts are intentionally simple:
- allocate A100s
- print `hostname` / `nvidia-smi`
- `sleep` for the reservation duration

## 9. What to do next in a fresh conversation
Recommended next actions depend on goal.

### Goal A: continue the science question quickly
Use the successful 14B entropy models.
Suggested next step:
- evaluate or distill students from the 14B entropy teachers
- compare against baseline 14B/non-entropy teacher pipeline

### Goal B: keep trying 32B
Only do this if there is a strong reason to insist on 32B.
Current evidence suggests the next realistic path is not more micro-adjustments to the same FSDP setup, but changing infra, e.g.:
- DeepSpeed / ZeRO
- more aggressive offload
- different sharding path
- larger resources

### Goal C: resume OlymMATH collection
Check remote status under:
- `/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/olymath_200_20250511_test`

Last known useful state:
- `Qwen3-32B` seeds `42..51` complete
- `seed52` at `98/200`
- `Qwen3-14B` not started

## 10. Minimal commands that were repeatedly useful

### Check active training/eval processes on a compute node
```bash
ps -ef | grep -E 'score_answers|filter_agent_training_data|torchrun|finetune_sft|run_agent_python_only_data_sequence|serve_vllm|run_experiment' | grep -v grep
```

### Check GPU usage
```bash
nvidia-smi
```

### Check main 32B entropy launcher log
```bash
tail -n 200 /scratch/wzhao20/AgentDistill/logs/train_teacher_entropy_32b_only_launcher.log
```

### Check 32B entropy training log
```bash
tail -n 200 /scratch/wzhao20/AgentDistill/logs/train_teacher_entropy_32b_lambda0p2.log
```

### Pull latest code on Hopper login node
```bash
cd /scratch/wzhao20/AgentDistill
/usr/bin/git pull --ff-only
```

## 11. Short final summary
- AgentDistill native eval/generation/training pipeline is in place and largely working.
- GSM-hard, MATH, and AIME teacher collection for 32B/14B are essentially done; OlymMATH is partially done.
- Native student distillation succeeded for six students.
- Entropy self-finetune was implemented as random-trajectory CE plus entropy regularization.
- 14B entropy teachers successfully trained for lambda 0.2 / 0.5 / 1.0.
- 32B entropy teacher repeatedly fails with rank `SIGKILL` during/just after checkpoint shard loading under the current 4xA100 FSDP route, even after aggressive reductions.
- The most reliable launch method on Hopper is currently: reserve with a hold job, then SSH directly into the allocated node and run the task there.
