"""
train_combined_rl.py — Combined OS-RL + Div-OBS-RL for anti-distillation.

Aggressive variant that maximises anti-distillation effect by combining two
complementary reward signals with stronger weights:

    R_total = R_task
            + lambda_sens    * R_sens       (observation-sensitivity)
            + lambda_div_obs * R_div_obs    (per-observation thought diversity)

R_sens    → makes each thought tightly bound to the specific execution result
R_div_obs → makes thoughts maximally diverse conditioned on the same obs position

Together they create a double bind for the student:
  - R_sens:    student must see the exact execution value to reconstruct the thought
  - R_div_obs: even if student sees the value, teacher's responses are inconsistent

Aggressive hyperparameters (vs. individual methods):
  - lambda_sens    = 0.3   (vs 0.1 in OS-RL)
  - lambda_div_obs = 1.0   (vs 0.5 in Div-OBS-RL)
  - lr             = 3e-5  (vs 1e-5)
  - kl_coeff       = 0.001 (vs 0.01 — near-unconstrained, allows more divergence)

Execution (4-GPU, DeepSpeed ZeRO-3):
    accelerate launch --config_file exps_research/mp_configs/accel_ds3_4gpu.yaml \\
        -m exps_research.rl_training.train_combined_rl \\
        --trajectory_dir  /scratch/.../evaluations \\
        --seed_range      42 57 \\
        --model_name      Qwen/Qwen3-32B \\
        --output_dir      training_outputs/qwen3-32B/combined_rl
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import random
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from accelerate import Accelerator
from torch.optim import AdamW

from .data_pool import TrajectoryPool
from .rewards import (
    compute_r_task,
    compute_r_sens,
    compute_r_div_obs,
)
from .grpo_trainer import GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_total_rewards_combined(
    model, tokenizer, group, lambda_sens, lambda_div_obs, device
):
    """
    R_total = R_task + lambda_sens * R_sens + lambda_div_obs * R_div_obs
    Returns (r_total, r_task, r_sens, r_div_obs)
    """
    r_task     = compute_r_task(group)
    r_sens     = compute_r_sens(model, tokenizer, group, device=device)
    r_div_obs  = compute_r_div_obs(group)
    r_total = [
        rt + lambda_sens * rs + lambda_div_obs * rd
        for rt, rs, rd in zip(r_task, r_sens, r_div_obs)
    ]
    return r_total, r_task, r_sens, r_div_obs


def glob_scored_files(trajectory_dir: str, seed_list: List[int]) -> List[str]:
    d = Path(trajectory_dir)
    files = []
    for seed in seed_list:
        matches = sorted(d.glob(f"*seed={seed}_*_scored.jsonl"))
        files.extend([str(m) for m in matches])
    if not files:
        files = [str(p) for p in sorted(d.glob("*_scored.jsonl"))]
    logger.info(f"Found {len(files)} scored trajectory files.")
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

    Root cause of previous crashes: get_peft_model_state_dict() calls model.state_dict()
    internally, which triggers an ALLREDUCE on rank 0 while ranks 1/2/3 are waiting for
    the GatheredParameters exit-BROADCAST — collective type mismatch → 30-min NCCL timeout.

    Fix: modifier_rank=None (no post-context broadcast) + build state dict directly from
    tensor.data (pure local read, no collective).
    """
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step}")
    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        import deepspeed, json as _json
        lora_params = [p for n, p in unwrapped.named_parameters() if "lora_" in n]
        param_dict = {}
        with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=None):
            if accelerator.is_main_process:
                param_dict = {
                    name: param.data.detach().cpu().clone()
                    for name, param in unwrapped.named_parameters()
                    if "lora_" in name
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


def resample_trajectories(args, ckpt_dir: str, output_dir: str, step: int) -> List[str]:
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
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb  = torch.cuda.memory_reserved()  / 1e9
        logger.info(f"GPU memory after prepare: allocated={allocated_gb:.1f}GB reserved={reserved_gb:.1f}GB")

    seed_list = list(range(args.seed_range[0], args.seed_range[1]))
    traj_files = glob_scored_files(args.trajectory_dir, seed_list)
    pool = TrajectoryPool(traj_files, min_group_size=args.min_group_size)

    if is_main:
        logger.info(f"Pool stats: {pool.stats()}")

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
    last_resample_step = global_step  # avoid immediate checkpoint on resume

    logger.info(f"Starting Combined-RL (aggressive) training loop (initial_step={global_step})...")
    logger.info(f"  lambda_sens={args.lambda_sens}, lambda_div_obs={args.lambda_div_obs}, "
                f"lr={args.lr}, kl_coeff={args.kl_coeff}")

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        for group in pool.iter_all_questions(
            n_trajs_per_question=args.n_trajs_per_question, shuffle=True
        ):
            if global_step >= args.max_steps:
                break

            # === Combined rewards: R_sens needs model forward (no_grad) ===
            r_total, r_task, r_sens, r_div_obs = compute_total_rewards_combined(
                model=model,
                tokenizer=tokenizer,
                group=group,
                lambda_sens=args.lambda_sens,
                lambda_div_obs=args.lambda_div_obs,
                device=accelerator.device,
            )

            # Free reward computation memory before GRPO forward
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                    "r_task_mean":    float(sum(r_task)    / max(len(r_task), 1)),
                    "r_sens_mean":    float(sum(r_sens)    / max(len(r_sens), 1)),
                    "r_div_obs_mean": float(sum(r_div_obs) / max(len(r_div_obs), 1)),
                    "r_total_mean":   float(sum(r_total)   / max(len(r_total), 1)),
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
                ckpt_dir = save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
                if args.do_resample:
                    new_files = resample_trajectories(
                        args, ckpt_dir, args.output_dir, global_step
                    )
                    if new_files:
                        pool.refresh(new_files)
                        logger.info(f"Pool refreshed. Stats: {pool.stats()}")

        if global_step >= args.max_steps:
            break

    save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
    if is_main:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Combined OS-RL + Div-OBS-RL (aggressive)")

    # Data
    p.add_argument("--trajectory_dir", type=str, required=True)
    p.add_argument("--seed_range", type=int, nargs=2, default=[42, 57])
    p.add_argument("--data_path", type=str,
                   default="data_processor/math_dataset/test/math_500_20250414.json")
    p.add_argument("--min_group_size", type=int, default=2)
    p.add_argument("--n_trajs_per_question", type=int, default=8)

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--max_length", type=int, default=2048)

    # Reward — aggressive defaults
    p.add_argument("--lambda_sens",    type=float, default=0.3,
                   help="Weight of R_sens (aggressive: 0.3 vs 0.1 in OS-RL)")
    p.add_argument("--lambda_div_obs", type=float, default=1.0,
                   help="Weight of R_div_obs (aggressive: 1.0 vs 0.5 in Div-OBS-RL)")

    # Training — aggressive defaults
    p.add_argument("--lr",           type=float, default=3e-5,
                   help="Learning rate (aggressive: 3e-5 vs 1e-5)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs",   type=int,   default=3)
    p.add_argument("--max_steps",    type=int,   default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--kl_coeff",     type=float, default=0.001,
                   help="KL penalty (aggressive: 0.001 vs 0.01 — near-unconstrained)")
    p.add_argument("--clip_ratio",   type=float, default=0.2)
    p.add_argument("--max_grad_norm",type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)

    # Resampling
    p.add_argument("--resample_every",  type=int,       default=200)
    p.add_argument("--do_resample",     action="store_true")
    p.add_argument("--resample_seeds",  type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--max_agent_steps", type=int,       default=5)

    # Output
    p.add_argument("--output_dir", type=str,
                   default="training_outputs/qwen3-32B/combined_rl")
    p.add_argument("--log_every",  type=int, default=10)

    # Checkpoint resuming
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to saved LoRA checkpoint directory to resume from")
    p.add_argument("--initial_step", type=int, default=0,
                   help="Starting step counter when resuming (set to checkpoint step)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
