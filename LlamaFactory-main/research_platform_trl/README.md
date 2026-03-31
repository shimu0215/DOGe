# Research Platform (smolagents + CodeAct + TRL)

This folder is a new, standalone experiment platform. It does **not** modify existing LlamaFactory workflows.

## Goals
- Unified CodeAct prompt and runtime for generation/eval and training data construction.
- Three experiment runners:
  1. `generate_eval_runner`: generate trajectories + evaluate first-sample accuracy.
  2. `kd_train_runner`: basic KD via trajectory SFT for student.
  3. `teacher_ft_runner`: teacher fine-tuning (current: SFT baseline; custom loss/RL hook ready in trainer).
- Stage-level resume checkpoints for interruption recovery.
- Context compression controls:
  - enable/disable toggle
  - fixed token budget or auto budget from model context limit

## Prompt usage
- `generate_eval_runner` uses smolagents built-in `CodeAgent` prompt template (CodeAct-style), and appends a
  question-level instruction to each question before calling `agent.run(...)`.
- `kd_train_runner` / `teacher_ft_runner` still read the shared prompt file:
  - `research_platform_trl/prompts/codeact_v1.yaml`

## Stage-level resume
Each runner writes stage checkpoint state under:
- `<work_dir>/checkpoints/*.json`

Run with `--resume` to skip completed stages.

## Runner 1: Generate + Eval
Config template:
- `research_platform_trl/configs/generate_eval_example.yaml`

Run:
```bash
cd /scratch/wzhao20/llama_factory
python -m research_platform_trl.runners.generate_eval_runner \
  --config research_platform_trl/configs/generate_eval_example.yaml
```

Resume:
```bash
python -m research_platform_trl.runners.generate_eval_runner \
  --config research_platform_trl/configs/generate_eval_example.yaml \
  --resume
```

Outputs:
- `<work_dir>/records.jsonl`
- `<work_dir>/contexts.jsonl`
- `<work_dir>/summary.json`
- Agent switch:
  - `context.enable_rolling_memory_code_agent: true` (default) uses `RollingMemoryCodeAgent`
  - `context.enable_rolling_memory_code_agent: false` uses plain smolagents `CodeAgent`

## Runner 2: KD Train (student)
Config template:
- `research_platform_trl/configs/kd_train_example.yaml`

Run:
```bash
python -m research_platform_trl.runners.kd_train_runner \
  --config research_platform_trl/configs/kd_train_example.yaml
```

Resume:
```bash
python -m research_platform_trl.runners.kd_train_runner \
  --config research_platform_trl/configs/kd_train_example.yaml \
  --resume
```

Outputs:
- `<work_dir>/train_data/kd_train.jsonl`
- `<work_dir>/student_model/`
- `<work_dir>/summary.json`

## Runner 3: Teacher FT (SFT baseline)
Config template:
- `research_platform_trl/configs/teacher_ft_example.yaml`

Run:
```bash
python -m research_platform_trl.runners.teacher_ft_runner \
  --config research_platform_trl/configs/teacher_ft_example.yaml
```

Resume:
```bash
python -m research_platform_trl.runners.teacher_ft_runner \
  --config research_platform_trl/configs/teacher_ft_example.yaml \
  --resume
```

Outputs:
- `<work_dir>/train_data/teacher_ft_train.jsonl`
- `<work_dir>/teacher_model/`
- `<work_dir>/summary.json`

## Notes
- Current KD/Teacher-FT uses TRL SFT (`SFTTrainer`) with chat messages.
- For future custom loss and RL, extend:
  - `research_platform_trl/trainers/teacher_sft.py`

## DeepSpeed (HPC)
`kd_train_runner` and `teacher_ft_runner` now support DeepSpeed through the train config:

```yaml
train:
  ...
  deepspeed: deepspeed_zero3.json
```

- `deepspeed` accepts either:
  - an absolute path, or
  - a path relative to the yaml config file location.
- Example config file provided:
  - `research_platform_trl/configs/deepspeed_zero3.json`

Recommended launch pattern on multi-GPU nodes:

```bash
torchrun --nproc_per_node=4 -m research_platform_trl.runners.kd_train_runner \
  --config research_platform_trl/configs/kd_train_example.yaml
```
