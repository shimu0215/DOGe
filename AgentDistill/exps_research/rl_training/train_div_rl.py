"""
train_div_rl.py — Diversity-RL: Trajectory diversity RL for anti-distillation.

Fine-tunes teacher with:
    R_total = R_task + lambda_div * R_diversity

where R_diversity rewards trajectories that use approaches different from
the other trajectories in the same question group, causing SFT students to
receive contradictory training signal and fail to learn a coherent policy.

No model forward passes are needed for R_diversity (only code extraction +
edit distance), making this cheaper to compute than OS-RL.

Execution (4-GPU, DeepSpeed ZeRO-3):
    accelerate launch --config_file exps_research/mp_configs/accel_ds3.yaml \\
        exps_research/rl_training/train_div_rl.py \\
        --trajectory_dir  logs/qa_results_python_only_teacher/math_500_20250414_test/evaluations \\
        --seed_range      42 57 \\
        --model_name      Qwen/Qwen3-32B \\
        --output_dir      training_outputs/qwen3-32B/div_rl
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
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from accelerate import Accelerator
from torch.optim import AdamW

from .data_pool import TrajectoryPool
from .rewards import compute_total_rewards_div_rl
from .grpo_trainer import GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with train_os_rl — kept local to avoid circular imports)
# ---------------------------------------------------------------------------

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
        use_cache=False,  # required for gradient checkpointing
        low_cpu_mem_usage=True,  # avoid duplicating weights during loading
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
    # use_reentrant=True is required for DeepSpeed ZeRO-3 compatibility
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
                class _SetEncoder(_json.JSONEncoder):
                    def default(self, o):
                        if isinstance(o, set):
                            return sorted(o)
                        return super().default(o)
                _json.dump(peft_cfg.to_dict(), f, indent=2, cls=_SetEncoder)
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

    # DeepSpeed custom-loop requires batch size to be declared explicitly
    if (hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None):
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = 1

    model, optimizer = accelerator.prepare(model, optimizer)

    # Diagnostic: confirm ZeRO-3 CPU offloading is reducing GPU footprint
    if torch.cuda.is_available() and accelerator.is_main_process:
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

    logger.info(f"Starting Diversity-RL training loop (initial_step={global_step})...")

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        for group in pool.iter_all_questions(
            n_trajs_per_question=args.n_trajs_per_question, shuffle=True
        ):
            if global_step >= args.max_steps:
                break

            # === Compute rewards (no model forward pass needed) ===
            r_total, r_task, r_div = compute_total_rewards_div_rl(
                group=group,
                lambda_div=args.lambda_div,
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
                    "r_div_mean": float(sum(r_div) / max(len(r_div), 1)),
                    "r_total_mean": float(sum(r_total) / max(len(r_total), 1)),
                }
                logger.info(json.dumps(log_entry))
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # === Periodic checkpoint + resampling ===
            # NOTE: all ranks must call save_checkpoint (uses GatheredParameters + barriers).
            _ckpt_due = (
                args.resample_every > 0
                and (global_step == 1 or global_step % args.resample_every == 0)
                and global_step > last_resample_step
            )
            if _ckpt_due:
                last_resample_step = global_step
                ckpt_dir = save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
                if is_main and args.do_resample:
                    new_files = resample_trajectories(
                        args, ckpt_dir, args.output_dir, global_step
                    )
                    if new_files:
                        pool.refresh(new_files)
                        logger.info(f"Pool refreshed with {len(new_files)} new files. "
                                    f"Stats: {pool.stats()}")

        if global_step >= args.max_steps:
            break

    if global_step > last_resample_step:
        save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
    if is_main:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Diversity-RL training")

    # Data
    p.add_argument("--trajectory_dir", type=str, required=True)
    p.add_argument("--seed_range", type=int, nargs=2, default=[42, 57])
    p.add_argument("--data_path", type=str,
                   default="data_processor/math_dataset/test/math_500_20250414.json")
    p.add_argument("--min_group_size", type=int, default=2)
    p.add_argument("--n_trajs_per_question", type=int, default=8)

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--max_length", type=int, default=4096)

    # Reward
    p.add_argument("--lambda_div", type=float, default=0.5,
                   help="Weight of R_diversity in R_total")

    # Training
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--kl_coeff", type=float, default=0.01)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Resampling
    p.add_argument("--resample_every", type=int, default=200)
    p.add_argument("--do_resample", action="store_true")
    p.add_argument("--resample_seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--max_agent_steps", type=int, default=5)

    # Output
    p.add_argument("--output_dir", type=str,
                   default="training_outputs/qwen3-32B/div_rl")
    p.add_argument("--log_every", type=int, default=10)

    # Checkpoint resuming
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to saved LoRA checkpoint directory to resume from")
    p.add_argument("--initial_step", type=int, default=0,
                   help="Starting step counter when resuming (set to checkpoint step)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
