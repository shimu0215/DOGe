"""
train_os_rl_online_pilot.py — Semi-online OS-RL pilot.

Same as train_os_rl.py but with working periodic resampling:
  - Every `resample_every` steps: save LoRA checkpoint, launch collect_unit.sh
    (vLLM tp=1 on GPU 3) to collect fresh trajectories for the pilot questions,
    then refresh the training pool.
  - Checkpoint is also saved at the same cadence (resample_every doubles as
    checkpoint_every).

GPU layout during resampling:
  Training  : 4 processes on GPUs 0-3 (ZeRO-3 CPU offload, ~1 GB each)
  vLLM      : tp-size=1, CUDA_VISIBLE_DEVICES=3  (uses the ~79 GB free on GPU 3)

This avoids NCCL group conflicts because tp=1 vLLM needs no inter-GPU comms.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import random
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from accelerate import Accelerator
from torch.optim import AdamW

from .data_pool import TrajectoryPool
from .rewards import compute_total_rewards_os_rl
from .grpo_trainer import GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]   # AgentDistill/


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def glob_scored_files(trajectory_dir: str, seed_list: List[int]) -> List[str]:
    d = Path(trajectory_dir)
    files = []
    for seed in seed_list:
        matches = sorted(d.glob(f"*seed={seed}_*_scored.jsonl"))
        files.extend([str(m) for m in matches])
    if not files:
        files = [str(p) for p in sorted(d.glob("*seed=*_*.jsonl"))]
    logger.info(f"Found {len(files)} trajectory files.")
    return files


def build_model(args):
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    if getattr(args, "resume_from_checkpoint", None):
        from peft import PeftModel
        logger.info(f"Resuming LoRA adapter from: {args.resume_from_checkpoint}")
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    model.print_trainable_parameters()
    return model


def build_tokenizer(args):
    return AutoTokenizer.from_pretrained(
        args.model_name,
        pad_token="<|endoftext|>",
        padding_side="left",
        add_eos_token=True,
        trust_remote_code=True,
    )


def save_checkpoint(model, tokenizer, output_dir: str, step: int, accelerator=None) -> str:
    """Save LoRA adapter checkpoint under ZeRO-3 without collective mismatch.

    Crash history:
      v4: named_parameters() called inside GatheredParameters context → triggers
          allgather_before hooks on non-LoRA params → NCCL collective mismatch
      v5 (this): pre-collect (name, param) pairs BEFORE entering context so the
                 context body only accesses pre-collected references.
    """
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step}")
    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        import deepspeed, json as _json
        lora_named_params = [(n, p) for n, p in unwrapped.named_parameters()
                             if "lora_" in n]
        lora_params = [p for _, p in lora_named_params]
        param_dict = {}
        with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=None):
            if accelerator.is_main_process:
                param_dict = {
                    name: param.data.detach().cpu().clone()
                    for name, param in lora_named_params
                }
        if accelerator.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(param_dict, os.path.join(ckpt_dir, "adapter_model.bin"))
            peft_cfg = list(unwrapped.peft_config.values())[0]
            with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
                _json.dump(peft_cfg.to_dict(), f, indent=2)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Checkpoint saved: {ckpt_dir}")
        accelerator.wait_for_everyone()
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Checkpoint saved: {ckpt_dir}")
    return ckpt_dir


def resample_trajectories(args, ckpt_dir: str, step: int) -> List[str]:
    """
    Collect fresh trajectories using the current LoRA checkpoint via vLLM.

    Uses collect_unit.sh with:
      - tp-size=1 (single GPU, no NCCL — avoids conflict with training NCCL groups)
      - CUDA_VISIBLE_DEVICES=3 (GPU 3; ~79 GB free alongside ZeRO-3 ~1 GB shard)

    Output goes to {ckpt_dir}/qa_results/{dataset_name}_test/...jsonl
    as determined by result_jsonl_path() in scripts_modular/common.sh.
    """
    collect_script = str(_ROOT / "scripts_modular" / "collect_unit.sh")
    dataset_name = Path(args.pilot_question_json).stem   # e.g. "pilot_questions"
    model_name = Path(args.model_name).name              # e.g. "Qwen3-32B"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "3"}

    new_files = []
    for seed in args.resample_seeds:
        logger.info(f"Resampling seed={seed} from checkpoint: {ckpt_dir}")
        cmd = [
            "bash", collect_script,
            "--model-id",        args.model_name,
            "--lora-folder",     ckpt_dir,
            "--data-path",       args.pilot_question_json,
            "--seed",            str(seed),
            "--tp-size",         "1",
            "--n",               "1",
            "--max-steps",       str(args.max_agent_steps),
            "--parallel-workers","4",
            "--gpu-util",        "0.85",
            "--max-lora-rank",   str(args.lora_r),
            "--force-rerun",     "1",    # always collect fresh
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                cwd=str(_ROOT))
        if result.returncode != 0:
            logger.warning(f"collect_unit.sh failed seed={seed}:\n{result.stderr[:600]}")
            continue

        # Locate the output file using the same naming as result_jsonl_path()
        expected = (
            Path(ckpt_dir) / "qa_results"
            / f"{dataset_name}_test"
            / f"{model_name}_temp=0.7_n=1_seed={seed}_type=agent_steps={args.max_agent_steps}"
              f"_python_only_python_only_seed{seed}.jsonl"
        )
        if expected.exists():
            new_files.append(str(expected))
            logger.info(f"Collected: {expected}")
        else:
            # Fallback glob in case naming differs slightly
            found = list((Path(ckpt_dir) / "qa_results").glob(
                f"**/*seed={seed}*python_only_seed{seed}.jsonl"
            ))
            new_files.extend([str(p) for p in found])
            if found:
                logger.info(f"Found via glob: {found}")

    logger.info(f"Resampling produced {len(new_files)} new files at step {step}.")
    return new_files


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=None,
    )
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = build_tokenizer(args)
    model = build_model(args)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if (hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None):
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = 1

    model, optimizer = accelerator.prepare(model, optimizer)

    if torch.cuda.is_available() and is_main:
        alloc = torch.cuda.memory_allocated() / 1e9
        resv  = torch.cuda.memory_reserved()  / 1e9
        logger.info(f"GPU memory after prepare: allocated={alloc:.1f}GB reserved={resv:.1f}GB")

    # Initial pool from pre-collected offline data
    seed_list = list(range(args.seed_range[0], args.seed_range[1]))
    traj_files = glob_scored_files(args.trajectory_dir, seed_list)
    pool = TrajectoryPool(traj_files, min_group_size=args.min_group_size)
    if is_main:
        logger.info(f"Initial pool: {pool.stats()}")

    grpo = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        accelerator=accelerator,
        config={
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "kl_coeff": args.kl_coeff,
            "max_length": args.max_length,
            "clip_ratio": args.clip_ratio,
            "max_grad_norm": args.max_grad_norm,
        },
    )

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    global_step = getattr(args, "initial_step", 0)
    optimizer_step = 0
    last_resample_step = global_step

    logger.info(f"Starting semi-online OS-RL pilot (initial_step={global_step})...")

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        for group in pool.iter_all_questions(
            n_trajs_per_question=args.n_trajs_per_question, shuffle=True
        ):
            if global_step >= args.max_steps:
                break

            r_total, r_task, r_sens = compute_total_rewards_os_rl(
                model=model,
                tokenizer=tokenizer,
                group=group,
                lambda_sens=args.lambda_sens,
                device=accelerator.device,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            info = grpo.step(
                batch_groups=[group],
                rewards_per_group=[r_total],
                global_step=global_step,
            )

            global_step += 1
            if info["optimizer_step"]:
                optimizer_step += 1

            if is_main and global_step % args.log_every == 0:
                log_entry = {
                    "step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": info.get("loss", 0.0),
                    "r_task_mean": float(sum(r_task) / max(len(r_task), 1)),
                    "r_sens_mean": float(sum(r_sens) / max(len(r_sens), 1)),
                    "r_total_mean": float(sum(r_total) / max(len(r_total), 1)),
                }
                logger.info(json.dumps(log_entry))
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # Checkpoint + online resampling
            if (
                is_main
                and args.resample_every > 0
                and global_step % args.resample_every == 0
                and global_step > last_resample_step
            ):
                last_resample_step = global_step
                ckpt_dir = save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
                new_files = resample_trajectories(args, ckpt_dir, global_step)
                if new_files:
                    pool.refresh(new_files)
                    logger.info(f"Pool refreshed with {len(new_files)} new files. "
                                f"Stats: {pool.stats()}")

        if global_step >= args.max_steps:
            break

    save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
    if is_main:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Semi-online OS-RL pilot")

    # Data
    p.add_argument("--trajectory_dir",     type=str, required=True,
                   help="Directory with offline pre-collected scored JSONL files")
    p.add_argument("--pilot_question_json",type=str, required=True,
                   help="JSON file with pilot questions (MATH-500 format) for resampling")
    p.add_argument("--seed_range",         type=int, nargs=2, default=[42, 57])
    p.add_argument("--min_group_size",     type=int, default=2)
    p.add_argument("--n_trajs_per_question",type=int,default=8)

    # Model
    p.add_argument("--model_name",   type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--lora_r",       type=int, default=32)
    p.add_argument("--max_length",   type=int, default=2048)

    # Reward
    p.add_argument("--lambda_sens",  type=float, default=0.1)

    # Training
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs",   type=int,   default=5)
    p.add_argument("--max_steps",    type=int,   default=150)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--kl_coeff",     type=float, default=0.01)
    p.add_argument("--clip_ratio",   type=float, default=0.2)
    p.add_argument("--max_grad_norm",type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)

    # Resampling (also triggers checkpointing)
    p.add_argument("--resample_every",    type=int,   default=50,
                   help="Resample + checkpoint every N steps")
    p.add_argument("--resample_seeds",    type=int, nargs="+", default=[42, 43, 44],
                   help="Seeds to use when collecting fresh trajectories")
    p.add_argument("--max_agent_steps",   type=int,   default=5)

    # Checkpoint resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--initial_step",       type=int,   default=0)

    # Output
    p.add_argument("--output_dir",   type=str,
                   default="training_outputs/qwen3-32B/os_rl_online_pilot")
    p.add_argument("--log_every",    type=int,   default=5)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
