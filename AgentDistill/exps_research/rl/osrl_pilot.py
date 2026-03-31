"""
OS-RL Pilot: offline GRPO on pre-collected trajectories.

Uses already-collected JSONL rollout files (no vLLM needed in this stage).
Validates:
  1. Sensitivity reward computation is numerically sensible
  2. GRPO loss back-propagation runs without OOM on 4×A100 80G
  3. LoRA parameters update correctly

Launch:
    torchrun --nproc_per_node=4 --master_port=29501 \
        -m exps_research.rl.osrl_pilot \
        --rollout_dir  /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher \
        --pattern      "math_500_20250414_test/Qwen3-32B_temp=0.7_seed=57*.jsonl" \
        --output_dir   /scratch/wzhao20/AgentDistill/training_outputs/osrl/pilot \
        --model_name   Qwen/Qwen3-32B \
        --n_problems   16  \
        --lambda_sensitivity 0.1 \
        --lora_rank    16  \
        --max_seq_length 4096
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from exps_research.rl.osrl_reward import compute_rewards

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("osrl_pilot")

_MASKED_LABEL = -100


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg, level="info"):
    if is_main():
        getattr(logger, level)(msg)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_text(content) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_trajectories(rollout_dir: str, pattern: str, n: int) -> List[Dict]:
    full_pattern = os.path.join(rollout_dir, pattern)
    files = glob.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"No files matched: {full_pattern}")
    log(f"Found {len(files)} rollout file(s): {files[:3]}")

    trajs: List[Dict] = []
    for fpath in sorted(files):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajs.append(json.loads(line))
        if len(trajs) >= n:
            break
    trajs = trajs[:n]
    log(f"Loaded {len(trajs)} trajectories (requested {n})")
    return trajs


# --------------------------------------------------------------------------- #
# Tokenisation for GRPO loss
# --------------------------------------------------------------------------- #

_ASSISTANT_ROLES = {"assistant", "tool-call"}
_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool-call": "assistant",
    "tool-response": "user",
}


def tokenise_trajectory(
    traj: Dict,
    tokenizer,
    max_length: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns (input_ids, labels) or None.
    labels[i] = token_id  for assistant / tool-call tokens
    labels[i] = -100      otherwise
    """
    messages = (traj.get("log_data") or {}).get("messages", [])
    if not messages:
        return None

    flat: List[Dict] = []
    is_action: List[bool] = []
    for msg in messages:
        role = _ROLE_MAP.get(msg["role"], "user")
        text = _get_text(msg["content"])
        if flat and flat[-1]["role"] == role:
            flat[-1]["content"] += "\n" + text
            is_action[-1] = is_action[-1] or (msg["role"] in _ASSISTANT_ROLES)
        else:
            flat.append({"role": role, "content": text})
            is_action.append(msg["role"] in _ASSISTANT_ROLES)

    try:
        full_ids = tokenizer.apply_chat_template(
            flat, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        )[0]
    except Exception as e:
        log(f"tokenise_trajectory error: {e}", "warning")
        return None

    if full_ids.shape[0] > max_length:
        return None

    labels = torch.full_like(full_ids, _MASKED_LABEL)
    cursor = 0
    for i, msg in enumerate(flat):
        try:
            prefix_ids = tokenizer.apply_chat_template(
                flat[: i + 1], tokenize=True, add_generation_prompt=False, return_tensors="pt"
            )[0]
            end = prefix_ids.shape[0]
            if is_action[i]:
                labels[cursor:end] = full_ids[cursor:end]
            cursor = end
        except Exception:
            pass

    return full_ids, labels


# --------------------------------------------------------------------------- #
# GRPO loss
# --------------------------------------------------------------------------- #

def compute_grpo_loss(
    policy,
    ref_model,
    tokenizer,
    trajectories: List[Dict],
    rewards: List[float],
    grpo_eps: float,
    kl_coef: float,
    max_seq_length: int,
    device: torch.device,
    local_rank: int,
) -> Tuple[torch.Tensor, Dict]:
    # Since offline pilot has 1 rollout per question, group size G=1
    # → advantage = 0 for all → no PG signal (expected).
    # For a proper pilot we duplicate trajectories with reward perturbation
    # or treat the whole batch as one group.
    # Here we treat the whole batch as one group.

    r_arr = np.array(rewards, dtype=np.float32)
    mean_r = r_arr.mean()
    std_r = r_arr.std() + grpo_eps
    advantages = (r_arr - mean_r) / std_r

    pg_acc   = torch.tensor(0.0, device=device)
    kl_acc   = torch.tensor(0.0, device=device)
    n_valid  = 0
    n_tokens = 0

    policy.train()
    for traj, adv in zip(trajectories, advantages):
        tok = tokenise_trajectory(traj, tokenizer, max_seq_length)
        if tok is None:
            continue
        input_ids, labels = tok
        input_ids = input_ids.unsqueeze(0).to(device)
        labels    = labels.unsqueeze(0).to(device)

        logits = policy(input_ids=input_ids).logits
        s_logits  = logits[:, :-1, :]
        s_labels  = labels[:, 1:]
        mask      = s_labels != _MASKED_LABEL

        if not mask.any():
            continue

        log_probs = F.log_softmax(s_logits.float(), dim=-1)
        tlps = log_probs.gather(-1, s_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        action_lps = tlps[mask]

        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids).logits
            ref_lps = F.log_softmax(ref_logits[:, :-1, :].float(), dim=-1)
            ref_tlps = ref_lps.gather(-1, s_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            ref_action_lps = ref_tlps[mask]

        pg_acc  = pg_acc  - float(adv) * action_lps.sum()
        kl_acc  = kl_acc  + (action_lps - ref_action_lps).sum()
        n_valid += 1
        n_tokens += mask.sum().item()

    if n_valid == 0:
        dummy = sum(p.sum() for p in policy.parameters() if p.requires_grad) * 0.0
        return dummy, {"loss": 0.0, "pg": 0.0, "kl": 0.0, "n_valid": 0}

    loss = (pg_acc + kl_coef * kl_acc) / n_valid
    stats = {
        "loss":    loss.item(),
        "pg":      (pg_acc / n_valid).item(),
        "kl":      (kl_acc / n_valid).item(),
        "n_valid": n_valid,
        "n_tokens": n_tokens,
    }
    return loss, stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    # --- distributed init ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    set_seed(args.seed + local_rank)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- load trajectories (rank 0 only, then share via file) ---
    if is_main():
        trajs = load_trajectories(args.rollout_dir, args.pattern, args.n_problems)
    if dist.is_initialized():
        dist.barrier()
    if not is_main():
        trajs = []  # other ranks don't need data for this pilot

    # --- build models ---
    log(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading base model (this takes a few minutes)...")
    # Use device_map to shard 32B across all 4 GPUs automatically
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base, lora_cfg)
    policy.print_trainable_parameters()
    policy.gradient_checkpointing_enable()

    log("Loading reference model ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    log("Models loaded.")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1e9
        log(f"  GPU {i}: {mem:.1f} GB allocated")

    # --- reward computation (main rank only) ---
    if is_main() and trajs:
        log("Computing rewards ...")
        rewards, r_task, r_sens = compute_rewards(
            policy, tokenizer, trajs, device,
            lambda_sensitivity=args.lambda_sensitivity,
            sensitivity_max_steps=5,
        )
        log(f"  R_task mean       : {np.mean(r_task):.4f}")
        log(f"  R_sensitivity mean: {np.mean(r_sens):.4f}")
        log(f"  R_total mean      : {np.mean(rewards):.4f}")
        log(f"  R_sensitivity std : {np.std(r_sens):.4f}")

        # Save reward stats
        stats_path = os.path.join(args.output_dir, "reward_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "n_trajectories": len(trajs),
                "r_task_mean": float(np.mean(r_task)),
                "r_task_std":  float(np.std(r_task)),
                "r_sens_mean": float(np.mean(r_sens)),
                "r_sens_std":  float(np.std(r_sens)),
                "r_total_mean": float(np.mean(rewards)),
                "lambda": args.lambda_sensitivity,
            }, f, indent=2)
        log(f"Reward stats saved to {stats_path}")
    else:
        rewards = []

    if dist.is_initialized():
        dist.barrier()

    # --- GRPO training step ---
    if is_main() and trajs and rewards:
        log("Running GRPO forward+backward pass ...")
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=args.learning_rate,
        )
        optimizer.zero_grad()
        loss, stats = compute_grpo_loss(
            policy, ref_model, tokenizer,
            trajs, rewards,
            grpo_eps=1e-8,
            kl_coef=args.kl_coef,
            max_seq_length=args.max_seq_length,
            device=device,
            local_rank=local_rank,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        log(f"GRPO step done:")
        log(f"  loss     = {stats['loss']:.6f}")
        log(f"  pg_loss  = {stats['pg']:.6f}")
        log(f"  kl_loss  = {stats['kl']:.6f}")
        log(f"  grad_norm= {grad_norm:.4f}")
        log(f"  n_valid  = {stats['n_valid']}")
        log(f"  n_tokens = {stats['n_tokens']}")

        # Save LoRA checkpoint
        ckpt_dir = os.path.join(args.output_dir, "lora_pilot_step1")
        policy.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        log(f"LoRA checkpoint saved to {ckpt_dir}")

        # Final GPU stats
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1e9
            peak = torch.cuda.max_memory_allocated(i) / 1e9
            log(f"  GPU {i}: current={mem:.1f}GB  peak={peak:.1f}GB")

        log("Pilot complete.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-32B")
    parser.add_argument("--rollout_dir", required=True,
                        help="root dir containing rollout JSONL subdirectories")
    parser.add_argument("--pattern", required=True,
                        help="glob pattern relative to rollout_dir, e.g. "
                             "'math_500_20250414_test/Qwen3-32B_temp=0.7_seed=57*.jsonl'")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_problems", type=int, default=16,
                        help="number of trajectories to use")
    parser.add_argument("--lambda_sensitivity", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
