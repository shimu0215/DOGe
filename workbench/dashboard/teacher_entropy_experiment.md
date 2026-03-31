# Teacher Entropy Self-Finetune

Date: 2026-03-25

## Goal

Test whether adding entropy regularization during teacher self-finetuning can make subsequent distillation harder.

## Implementation

Repository:
- `/Users/shimu/Downloads/DOGe-main/AgentDistill`

Commit:
- `e473403` `add teacher entropy self-finetune mode`

Key changes:
- `exps_research/finetune_sft.py`
  - added `EntropyRegularizedSFTTrainer`
  - added `--use_entropy_regularization`
  - added `--entropy_lambda`
  - added `--random_trajectory_per_question`
- `exps_research/train_utils/preprocess.py`
  - added grouped trajectory loading by question
- `scripts/training/train_teacher_entropy_math_sequence.sh`
  - scores + filters `Qwen3-32B / MATH / seed42..56`
  - trains teacher self-finetune with grouped random trajectory sampling
  - default lambdas: `0.2 0.5 1.0`
  - default target model: `Qwen/Qwen3-32B`
  - fallback target if needed: `Qwen/Qwen3-14B`

## Intended Hopper flow

1. Let current `AIME/OlymMATH` collection finish on the active 4-GPU hold.
2. Pull commit `e473403` on Hopper.
3. Launch `scripts/training/train_teacher_entropy_math_sequence.sh` on the active 4-GPU node.
4. If `Qwen3-32B` fails to fit or train stably, rerun with:
   - `TARGET_MODEL=Qwen/Qwen3-14B`
   - same lambdas
5. Record progress and failures in the current Hopper status file.

## Loss

For a sampled teacher trajectory `y`:

- SFT term:
  - cross-entropy on the sampled next token
- Entropy term:
  - maximize token entropy on supervised response positions

Total objective:

`L = L_sft - lambda * H_token`

where `H_token` is the mean token entropy over non-masked labels.
