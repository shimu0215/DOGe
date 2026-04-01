"""
OS-RL Offline GRPO Training.

Uses pre-collected multi-seed trajectories (seeds 42-53, G=12 per problem)
to run GRPO without needing online vLLM rollout.

Saves a checkpoint every --save_every iterations so training can resume
on a different GPU allocation.

Launch (single process, device_map=auto across 4xA100):
    python -m exps_research.rl.osrl_offline_train \
        --data_dir  /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test \
        --seed_range 42 53 \
        --model_name Qwen/Qwen3-32B \
        --output_dir /scratch/wzhao20/AKDA2-vjk/AgentDistill/training_outputs/osrl/run1

Resume on a new GPU allocation:
    python -m exps_research.rl.osrl_offline_train \
        ... same args ... \
        --resume_from_checkpoint <output_dir>/checkpoints/iter0030
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from exps_research.rl.osrl_reward import compute_rewards

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("osrl_train")

_MASKED_LABEL = -100
_ASSISTANT_ROLES = {"assistant", "tool-call"}
_ROLE_MAP = {
    "system": "system", "user": "user",
    "assistant": "assistant", "tool-call": "assistant",
    "tool-response": "user",
}


# =========================================================================== #
# Helpers
# =========================================================================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_text(content) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def gpu_mem_str() -> str:
    lines = []
    for i in range(torch.cuda.device_count()):
        cur  = torch.cuda.memory_allocated(i) / 1e9
        peak = torch.cuda.max_memory_allocated(i) / 1e9
        lines.append(f"GPU{i}: {cur:.1f}/{peak:.1f}GB")
    return "  ".join(lines)


# =========================================================================== #
# Data loading
# =========================================================================== #

def load_grouped_trajectories(
    data_dir: str,
    seed_start: int,
    seed_end: int,
    model_prefix: str = "Qwen3-32B",
) -> Dict[str, List[Dict]]:
    """
    Load JSONL files for seeds [seed_start, seed_end] and group by question.
    Returns: { question_text: [traj1, traj2, ...] }  (G trajectories each)
    """
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    total = 0
    for seed in range(seed_start, seed_end + 1):
        pattern = os.path.join(
            data_dir,
            f"{model_prefix}_temp=0.7_seed={seed}_type=agent_steps=5_python_only_python_only_seed{seed}.jsonl",
        )
        files = glob.glob(pattern)
        if not files:
            logger.warning(f"No file for seed {seed}: {pattern}")
            continue
        with open(files[0]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traj = json.loads(line)
                # Some entries are stored as JSON strings containing Python dict
                # reprs (single-quoted). Use ast.literal_eval to recover the dict.
                if isinstance(traj, str):
                    try:
                        traj = ast.literal_eval(traj)
                    except Exception:
                        logger.warning("Skipping unparseable line")
                        continue
                if not isinstance(traj, dict):
                    continue
                q = str(traj.get("question", ""))
                grouped[q].append(traj)
                total += 1
    logger.info(
        f"Loaded {total} trajectories across {len(grouped)} questions "
        f"(seeds {seed_start}-{seed_end})"
    )
    return dict(grouped)


# =========================================================================== #
# Trajectory tokenisation
# =========================================================================== #

def _apply_tmpl(tokenizer, msgs, add_gen: bool = False) -> torch.Tensor:
    out = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=add_gen, return_tensors="pt"
    )
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return out.squeeze(0) if out.dim() == 2 else out


def tokenise_trajectory(
    traj: Dict,
    tokenizer,
    max_length: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Returns (input_ids, labels) or None."""
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
        full_ids = _apply_tmpl(tokenizer, flat)
    except Exception as e:
        logger.debug(f"tokenise error: {e}")
        return None

    if full_ids.shape[0] > max_length:
        return None

    labels = torch.full_like(full_ids, _MASKED_LABEL)
    cursor = 0
    for i in range(len(flat)):
        try:
            prefix_ids = _apply_tmpl(tokenizer, flat[: i + 1])
            end = prefix_ids.shape[0]
            if is_action[i]:
                labels[cursor:end] = full_ids[cursor:end]
            cursor = end
        except Exception:
            pass

    return full_ids, labels


# =========================================================================== #
# GRPO loss
# =========================================================================== #

def grpo_loss_for_group(
    policy,
    ref_model,
    tokenizer,
    group_trajs: List[Dict],
    group_rewards: List[float],
    kl_coef: float,
    clip_eps: float,
    grpo_eps: float,
    max_seq_length: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """Compute GRPO loss for one question's G rollouts."""
    r = np.array(group_rewards, dtype=np.float32)
    advantages = (r - r.mean()) / (r.std() + grpo_eps)

    pg_acc = torch.tensor(0.0, device=device)
    kl_acc = torch.tensor(0.0, device=device)
    n_valid = 0

    for traj, adv in zip(group_trajs, advantages):
        tok = tokenise_trajectory(traj, tokenizer, max_seq_length)
        if tok is None:
            continue
        input_ids, labels = tok
        input_ids = input_ids.unsqueeze(0).to(device)
        labels    = labels.unsqueeze(0).to(device)

        logits = policy(input_ids=input_ids).logits
        s_logits = logits[:, :-1, :]
        s_labels = labels[:, 1:]
        mask     = s_labels != _MASKED_LABEL
        if not mask.any():
            continue

        lp = F.log_softmax(s_logits.float(), dim=-1)
        tlp = lp.gather(-1, s_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        act_lp = tlp[mask]

        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids).logits
            ref_lp = F.log_softmax(ref_logits[:, :-1, :].float(), dim=-1)
            ref_tlp = ref_lp.gather(-1, s_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            ref_act_lp = ref_tlp[mask]

        pg_acc = pg_acc - float(adv) * act_lp.sum()
        kl_acc = kl_acc + (act_lp - ref_act_lp).sum()
        n_valid += 1

    if n_valid == 0:
        return None, {"n_valid": 0}

    loss = (pg_acc + kl_coef * kl_acc) / n_valid
    return loss, {
        "pg": (pg_acc / n_valid).item(),
        "kl": (kl_acc / n_valid).item(),
        "n_valid": n_valid,
    }


# =========================================================================== #
# Checkpoint helpers
# =========================================================================== #

def save_checkpoint(policy, tokenizer, optimizer, scheduler, iteration: int, output_dir: str) -> str:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"iter{iteration:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    # Save optimizer + scheduler state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
    }, os.path.join(ckpt_dir, "train_state.pt"))
    logger.info(f"Checkpoint saved → {ckpt_dir}  [{gpu_mem_str()}]")
    return ckpt_dir


def load_checkpoint(policy, tokenizer, optimizer, scheduler, ckpt_dir: str) -> int:
    """Load from checkpoint; returns the iteration number."""
    # Load LoRA weights
    policy.load_adapter(ckpt_dir, adapter_name="default")
    state_path = os.path.join(ckpt_dir, "train_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        iteration = state["iteration"]
        logger.info(f"Resumed from {ckpt_dir} at iteration {iteration}")
        return iteration
    return 0


# =========================================================================== #
# Main training loop
# =========================================================================== #

def train(args) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # ---- load data ----
    grouped = load_grouped_trajectories(
        args.data_dir, args.seed_start, args.seed_end
    )
    questions = list(grouped.keys())
    logger.info(f"Training on {len(questions)} questions, "
                f"G={max(len(v) for v in grouped.values())} rollouts/question")

    # ---- build model ----
    device = torch.device("cuda:0")  # device_map=auto handles the rest
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading policy model ...")
    if args.resume_from_checkpoint and os.path.exists(
        os.path.join(args.resume_from_checkpoint, "adapter_config.json")
    ):
        logger.info(f"  Loading LoRA adapter from {args.resume_from_checkpoint}")
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        policy = PeftModel.from_pretrained(base, args.resume_from_checkpoint, is_trainable=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base, lora_cfg)

    policy.gradient_checkpointing_enable()
    policy.print_trainable_parameters()

    logger.info("Loading reference model ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    logger.info(f"Models loaded.  {gpu_mem_str()}")

    # ---- optimizer & scheduler ----
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.num_iterations * args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    # ---- resume ----
    start_iter = 0
    if args.resume_from_checkpoint:
        train_state = os.path.join(args.resume_from_checkpoint, "train_state.pt")
        if os.path.exists(train_state):
            state = torch.load(train_state, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_iter = state["iteration"]
            logger.info(f"Resumed at iteration {start_iter}")

    # ---- training loop ----
    total_loss_acc  = 0.0
    total_pg_acc    = 0.0
    total_kl_acc    = 0.0
    total_rt_acc    = 0.0
    total_rs_acc    = 0.0
    n_groups_acc    = 0
    optimizer.zero_grad()

    for iteration in range(start_iter, args.num_iterations):
        t0 = time.time()

        # Sample batch of questions
        batch_qs = random.sample(questions, min(args.batch_size, len(questions)))

        # Gather trajectories
        batch_trajs: List[Dict]  = []
        batch_rewards: List[float] = []
        batch_groups: List[Tuple[int, int]] = []  # (start, end) in batch_trajs

        policy.eval()
        for q in batch_qs:
            rollouts = grouped[q]
            # Down-sample to args.g_per_problem if we have more
            if len(rollouts) > args.g_per_problem:
                rollouts = random.sample(rollouts, args.g_per_problem)

            rewards, r_task, r_sens = compute_rewards(
                policy, tokenizer, rollouts, device,
                lambda_sensitivity=args.lambda_sensitivity,
                sensitivity_max_steps=args.sensitivity_max_steps,
            )
            start = len(batch_trajs)
            batch_trajs.extend(rollouts)
            batch_rewards.extend(rewards)
            batch_groups.append((start, start + len(rollouts)))

            total_rt_acc += float(np.mean(r_task))
            total_rs_acc += float(np.mean(r_sens))

        # GRPO update
        policy.train()
        iter_loss = 0.0
        iter_pg   = 0.0
        iter_kl   = 0.0
        n_groups  = 0

        for (start, end) in batch_groups:
            g_trajs   = batch_trajs[start:end]
            g_rewards = batch_rewards[start:end]

            loss, stats = grpo_loss_for_group(
                policy, ref_model, tokenizer,
                g_trajs, g_rewards,
                kl_coef=args.kl_coef,
                clip_eps=args.clip_eps,
                grpo_eps=1e-8,
                max_seq_length=args.max_seq_length,
                device=device,
            )
            if loss is None:
                continue

            (loss / args.grad_accum_steps).backward()
            iter_loss += loss.item()
            iter_pg   += stats["pg"]
            iter_kl   += stats["kl"]
            n_groups  += 1

        if n_groups > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

            # Step every grad_accum_steps iterations
            if (iteration + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss_acc += iter_loss / n_groups
            total_pg_acc   += iter_pg   / n_groups
            total_kl_acc   += iter_kl   / n_groups
            n_groups_acc   += 1

        elapsed = time.time() - t0
        n_done = iteration - start_iter + 1

        if (iteration + 1) % args.log_every == 0 and n_groups_acc > 0:
            avg_loss = total_loss_acc / n_groups_acc
            avg_pg   = total_pg_acc   / n_groups_acc
            avg_kl   = total_kl_acc   / n_groups_acc
            avg_rt   = total_rt_acc   / (n_done * args.batch_size)
            avg_rs   = total_rs_acc   / (n_done * args.batch_size)
            lr_now   = scheduler.get_last_lr()[0]
            logger.info(
                f"iter {iteration+1:04d}/{args.num_iterations}  "
                f"loss={avg_loss:.4f}  pg={avg_pg:.4f}  kl={avg_kl:.6f}  "
                f"R_task={avg_rt:.3f}  R_sens={avg_rs:.3f}  "
                f"grad_norm={grad_norm:.2f}  lr={lr_now:.2e}  "
                f"t={elapsed:.0f}s  [{gpu_mem_str()}]"
            )
            # Reset accumulators
            total_loss_acc = total_pg_acc = total_kl_acc = 0.0
            total_rt_acc   = total_rs_acc = 0.0
            n_groups_acc   = 0

        # Checkpoint
        if (iteration + 1) % args.save_every == 0:
            save_checkpoint(policy, tokenizer, optimizer, scheduler,
                            iteration + 1, args.output_dir)

    # Final checkpoint
    logger.info("Training complete — saving final checkpoint ...")
    save_checkpoint(policy, tokenizer, optimizer, scheduler,
                    args.num_iterations, args.output_dir)


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",   default="Qwen/Qwen3-32B")
    p.add_argument("--data_dir",     required=True,
                   help="dir containing per-seed JSONL files")
    p.add_argument("--seed_start",   type=int, default=42)
    p.add_argument("--seed_end",     type=int, default=53)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--resume_from_checkpoint", default=None)

    # LoRA
    p.add_argument("--lora_rank",    type=int, default=32)

    # Reward
    p.add_argument("--lambda_sensitivity",  type=float, default=0.1)
    p.add_argument("--sensitivity_max_steps", type=int, default=2)

    # GRPO
    p.add_argument("--num_iterations",  type=int, default=200)
    p.add_argument("--batch_size",      type=int, default=16,
                   help="problems per iteration")
    p.add_argument("--g_per_problem",   type=int, default=8,
                   help="max rollouts per problem (≤ available seeds)")
    p.add_argument("--kl_coef",         type=float, default=0.01)
    p.add_argument("--clip_eps",        type=float, default=0.2)
    p.add_argument("--max_seq_length",  type=int,   default=6144)

    # Optimiser
    p.add_argument("--lr",              type=float, default=2e-6)
    p.add_argument("--grad_accum_steps",type=int,   default=4)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--warmup_ratio",    type=float, default=0.05)

    # Logging / saving
    p.add_argument("--log_every",   type=int, default=5)
    p.add_argument("--save_every",  type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    train(parse_args())
