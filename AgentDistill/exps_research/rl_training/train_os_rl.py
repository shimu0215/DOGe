"""
train_os_rl.py — OS-RL: Output-Sensitivity RL for anti-distillation.

Fine-tunes teacher with:
    R_total = R_task + lambda_sens * R_sens

where R_sens rewards trajectories whose intermediate reasoning is highly
dependent on specific execution results — making them hard to imitate.

Execution (4-GPU, DeepSpeed ZeRO-3):
    accelerate launch --config_file exps_research/mp_configs/accel_ds3.yaml \\
        exps_research/rl_training/train_os_rl.py \\
        --trajectory_dir  logs/qa_results_python_only_teacher/math_500_20250414_test/evaluations \\
        --seed_range      42 57 \\
        --model_name      Qwen/Qwen3-32B \\
        --output_dir      training_outputs/qwen3-32B/os_rl
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import random
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch.optim import AdamW

from .data_pool import TrajectoryPool
from .rewards import compute_total_rewards_os_rl
from .grpo_trainer import GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def glob_scored_files(trajectory_dir: str, seed_range: List[int]) -> List[str]:
    """Collect scored JSONL files for the given seed range."""
    d = Path(trajectory_dir)
    files = []
    for seed in seed_range:
        matches = sorted(d.glob(f"*seed={seed}_*_scored.jsonl"))
        files.extend([str(m) for m in matches])
    if not files:
        # Fallback: all scored files in dir
        files = [str(p) for p in sorted(d.glob("*_scored.jsonl"))]
    logger.info(f"Found {len(files)} scored trajectory files.")
    return files


def build_model(args, accelerator):
    """Load base model and wrap with LoRA."""
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        pad_token="<|endoftext|>",
        padding_side="left",
        add_eos_token=True,
        trust_remote_code=True,
    )
    return tokenizer


def save_checkpoint(model, tokenizer, output_dir: str, step: int):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info(f"Checkpoint saved: {ckpt_dir}")
    return ckpt_dir


def resample_trajectories(args, ckpt_dir: str, output_dir: str, step: int) -> List[str]:
    """
    Launch vLLM-based trajectory collection using the current checkpoint.
    Returns list of newly created scored JSONL paths.

    Calls the existing collect_unit.sh script for each seed in resample_seeds.
    The new trajectories are written to output_dir/resampled/step{step}/.
    """
    resample_out = os.path.join(output_dir, "resampled", f"step{step}")
    os.makedirs(resample_out, exist_ok=True)

    collect_script = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts_modular", "collect_unit.sh"
    )

    new_files = []
    for seed in args.resample_seeds:
        logger.info(f"Resampling: seed={seed}, checkpoint={ckpt_dir}")
        cmd = [
            "bash", collect_script,
            "--model_path", ckpt_dir,
            "--data_path", args.data_path,
            "--output_dir", resample_out,
            "--seed", str(seed),
            "--task_type", "math",
            "--max_steps", str(args.max_agent_steps),
            "--search_type", "python_only",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"collect_unit.sh failed (seed={seed}): {result.stderr[:500]}")
        else:
            # Look for scored files created in resample_out
            new = list(Path(resample_out).glob(f"*seed={seed}*_scored.jsonl"))
            new_files.extend([str(p) for p in new])

    logger.info(f"Resampling produced {len(new_files)} new scored files.")
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

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model & tokenizer
    tokenizer = build_tokenizer(args)
    model = build_model(args, accelerator)

    # Optimizer (only LoRA params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer = accelerator.prepare(model, optimizer)

    # Trajectory pool (seed range as closed interval)
    seed_list = list(range(args.seed_range[0], args.seed_range[1]))
    traj_files = glob_scored_files(args.trajectory_dir, seed_list)
    pool = TrajectoryPool(traj_files, min_group_size=args.min_group_size)

    if is_main:
        logger.info(f"Pool stats: {pool.stats()}")

    # GRPO trainer
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

    # Logging state
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    global_step = 0
    optimizer_step = 0
    last_resample_step = 0

    logger.info("Starting OS-RL training loop...")

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        for group in pool.iter_all_questions(
            n_trajs_per_question=args.n_trajs_per_question, shuffle=True
        ):
            if global_step >= args.max_steps:
                break

            # === Compute rewards (no_grad) ===
            # Only on main process for logging clarity; all ranks compute
            r_total, r_task, r_sens = compute_total_rewards_os_rl(
                model=model,
                tokenizer=tokenizer,
                group=group,
                lambda_sens=args.lambda_sens,
                device=accelerator.device,
            )

            # === GRPO step ===
            info = grpo.step(
                batch_groups=[group],
                rewards_per_group=[r_total],
                global_step=global_step,
            )

            global_step += 1
            if info["optimizer_step"]:
                optimizer_step += 1

            # === Logging ===
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

            # === Periodic checkpoint + resampling ===
            if (
                is_main
                and args.resample_every > 0
                and global_step % args.resample_every == 0
                and global_step > last_resample_step
            ):
                last_resample_step = global_step
                ckpt_dir = save_checkpoint(model, tokenizer, args.output_dir, global_step)
                if args.do_resample:
                    new_files = resample_trajectories(
                        args, ckpt_dir, args.output_dir, global_step
                    )
                    if new_files:
                        pool.refresh(new_files)
                        logger.info(f"Pool refreshed with {len(new_files)} new files. "
                                    f"Stats: {pool.stats()}")

        if global_step >= args.max_steps:
            break

    # Final checkpoint
    if is_main:
        save_checkpoint(model, tokenizer, args.output_dir, global_step)
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OS-RL: Output-Sensitivity RL training")

    # Data
    p.add_argument("--trajectory_dir", type=str, required=True,
                   help="Directory containing *_scored.jsonl files")
    p.add_argument("--seed_range", type=int, nargs=2, default=[42, 57],
                   help="[start, end) seeds to load (e.g. 42 57 → seeds 42-56)")
    p.add_argument("--data_path", type=str,
                   default="data_processor/math_dataset/test/math_500_20250414.json",
                   help="Question dataset path (for resampling)")
    p.add_argument("--min_group_size", type=int, default=2,
                   help="Min trajectories per question for GRPO grouping")
    p.add_argument("--n_trajs_per_question", type=int, default=8,
                   help="Trajectories per question in each GRPO group")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--max_length", type=int, default=4096)

    # Reward
    p.add_argument("--lambda_sens", type=float, default=0.1,
                   help="Weight of R_sens in R_total")

    # Training
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=1000,
                   help="Max gradient accumulation steps total")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--kl_coeff", type=float, default=0.01,
                   help="KL penalty coefficient (0 = disabled)")
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Resampling
    p.add_argument("--resample_every", type=int, default=200,
                   help="Resample trajectories every N global steps (0=disabled)")
    p.add_argument("--do_resample", action="store_true",
                   help="Actually launch collection script during resampling")
    p.add_argument("--resample_seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="Seeds to use when resampling trajectories")
    p.add_argument("--max_agent_steps", type=int, default=5,
                   help="Max agent steps during resampling collection")

    # Output
    p.add_argument("--output_dir", type=str,
                   default="training_outputs/qwen3-32B/os_rl")
    p.add_argument("--log_every", type=int, default=10)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
