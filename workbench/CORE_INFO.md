# Core Info Record

All AI working in this workspace must read and follow this file.

## Scope

- The shared-information system has been intentionally simplified.
- Do not recreate task logs, knowledge bases, or collaboration ledgers unless the user explicitly asks for them again.
- Keep workbench focused on:
  - AI registration
  - this core info record
  - the Hopper dashboard

## Project Purpose

- Student behavior:
  the student is SFT-trained on teacher trajectories, then at test time the student calls tools and solves tasks on its own.
- This project is for an intellectual-property protection setting:
  the teacher is a private model, and the goal is to prevent publicly exposed API trajectories from letting a third party distill a student with comparable capability.

## Main Goals

- Goal 1:
  after fine-tuning, the teacher's own performance should not go down.
- Goal 2:
  students trained from the fine-tuned teacher should perform much worse than students trained from the non-fine-tuned teacher.

## Code And Execution Rule

- All code changes must be made locally first.
- After local code changes, push them to the repository.
- On Hopper, pull the repository updates before running.
- The default execution flow is:
  1. edit locally
  2. push to repo
  3. `ssh hopper`
  4. pull on Hopper
  5. run on Hopper
- Unless the user explicitly asks for it, do not cancel Hopper jobs.
- If a running job needs to stop work, only kill the process inside the job instead of canceling the job itself.

## Dashboard Scope

- The dashboard should show only information obtained directly from Hopper.
- The dashboard should focus on:
  - reserved/running jobs
  - queued jobs
  - GPU type
  - on-demand GPU memory and utilization details
  - SSH / Duo status
- Do not rely on local task logs or knowledge logs for dashboard content.

## Collaboration Rule

- Other AI only need to do two things here:
  - register a unique name
  - read and obey this core info record

## Hopper Usage Note

- The Hopper connection path is already debugged through `ssh hopper`.
- After connecting, you can inspect reserved GPUs through the user's reservation jobs.
- These reservations are commonly held by long-running sleep/hold loops.
- Work is usually launched onto the reserved allocation with `srun` from the held job.
- Prefer preserving the reservation and running work inside it rather than tearing it down.

## Hopper Repo And Env

- The Hopper repo root is:
  - `/scratch/wzhao20/AKDA2`
- The AgentDistill runtime root is:
  - `/scratch/wzhao20/AKDA2/AgentDistill`
- The current clean Hopper conda environment is:
  - `/scratch/wzhao20/conda_envs/AKDA1`
- The verified core package versions in `AKDA1` are:
  - `torch = 2.8.0+cu128`
  - `transformers = 4.57.1`
  - `vllm = 0.11.0`
- `AKDA1` has already passed:
  - `pip check`
  - import checks for `torch`, `transformers`, `vllm`, `datasets`, `accelerate`, `deepspeed`, `trl`, and `PIL.Image`

## Hopper Env Variables

- When running AgentDistill on Hopper, prefer these defaults:
  - `CONDA_ENV_PREFIX=/scratch/wzhao20/conda_envs/AKDA1`
  - `PATH=/scratch/wzhao20/conda_envs/AKDA1/bin:$PATH`
  - `PYTHONPATH=/scratch/wzhao20/AKDA2/AgentDistill/src:${PYTHONPATH:-}`
  - `HF_HOME=/scratch/wzhao20/hf_cache`
  - `TRANSFORMERS_CACHE=$HF_HOME`
  - `HF_DATASETS_CACHE=$HF_HOME/datasets`
  - `XDG_CACHE_HOME=/scratch/wzhao20/.cache`
  - `VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache`
  - `TRITON_CACHE_DIR=/scratch/wzhao20/triton_cache`
  - `TORCHINDUCTOR_CACHE_DIR=/scratch/wzhao20/torchinductor_cache`
  - `VLLM_NO_USAGE_STATS=1`
  - `DO_NOT_TRACK=1`

## Modular Script Location

- The new modular AgentDistill scripts live at:
  - `/scratch/wzhao20/AKDA2/AgentDistill/scripts_modular`
- The intended roles are:
  - `collect_unit.sh`: one model + one dataset + one seed collection unit
  - `check_quality.sh`: optional quality inspection only when explicitly requested
  - `collect_batch.sh`: upper-level batch wrapper that calls `collect_unit.sh`
  - `common.sh`: shared environment setup and cleanup helpers

## Change Control

- If you change the dashboard behavior, keep it aligned with the direct-from-Hopper rule above.
- If you think this minimal system is insufficient, ask the user before reintroducing broader shared-memory structures.
