"""
train_grpo_online.py — True-online GRPO on MATH-500 with Qwen3-14B.

GPU layout (fixed split):
  GPU 0,1  CUDA_VISIBLE_DEVICES=0,1 — vLLM offline rollout (tp=2)
  GPU 2,3  CUDA_VISIBLE_DEVICES=2,3 — accelerate training (ZeRO-3, LoRA, 2 proc)

Reward per trajectory:
  R_total = R_task + λ_kl · KL(π_θ(·|c_t) ‖ π_ref(·|c_t))

  R_task  : 1.0 if final answer correct, 0.0 otherwise (from scored JSONL).
  KL term : mean_t( log π_θ(a_t|c_t) - log π_ref(a_t|c_t) )
            computed per-token over ALL assistant turns, averaged.
            π_ref = frozen base model (LoRA adapter disabled).
            λ_kl > 0 rewards trajectories that DIVERGE from the reference,
            aligning with the anti-distillation objective.

Online update cycle:
  Every `rollout_every` steps:
    1. Save current LoRA checkpoint.
    2. Spawn vLLM subprocess on GPU 0,1 (tp=2) to collect K fresh trajectories
       per question for the current batch of B questions.
    3. Replace online pool with these fresh trajectories (fully on-policy).
  Between rolloutes:
    4. For each question-group in the pool, compute R_total, run GRPO step.
  Every `checkpoint_every` steps: save checkpoint (if not already saved this step).

Prompt/data format:
  Identical to the existing smolagents CodeAct pipeline (run_experiment.py).
  Temperature > 0 ensures trajectory diversity within each K-copy rollout.

References:
  GRPO: Shao et al., 2024.  PPO-clip-KL: Schulman et al., 2017.
  Semi-online OS-RL: train_os_rl_online_pilot.py in this repo.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import random
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import timedelta, datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.optim import AdamW

from .data_pool import TrajectoryPool
from .rewards import compute_r_task
from .grpo_trainer import (
    GRPOTrainer,
    compute_trajectory_log_probs,
    _ref_log_prob,
)
from .message_utils import clean_messages_for_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]   # AgentDistill/


# ---------------------------------------------------------------------------
# Model / tokenizer builders
# ---------------------------------------------------------------------------

def build_model(args):
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    if args.resume_from_checkpoint:
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


# ---------------------------------------------------------------------------
# Checkpoint save  (ZeRO-3 safe — identical to train_os_rl_online_pilot.py)
# ---------------------------------------------------------------------------

def save_checkpoint(model, tokenizer, output_dir: str, step: int, accelerator) -> str:
    import deepspeed
    import json as _json
    import re as _re

    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step}")
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)

    lora_named_params = [(n, p) for n, p in unwrapped.named_parameters()
                         if "lora_" in n]
    lora_params = [p for _, p in lora_named_params]

    def _clean_lora_key(name: str) -> str:
        return _re.sub(
            r'\.(lora_A|lora_B|lora_embedding_A|lora_embedding_B)\.([^.]+)\.',
            r'.\1.',
            name,
        )

    with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=None):
        if accelerator.is_main_process:
            param_dict = {
                _clean_lora_key(n): p.data.detach().cpu().clone()
                for n, p in lora_named_params
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
    return ckpt_dir


# ---------------------------------------------------------------------------
# KL reward computation
# ---------------------------------------------------------------------------

def compute_kl_rewards(
    model,
    tokenizer,
    group: List[dict],
    device: torch.device,
    max_length: int = 4096,
) -> List[float]:
    """
    For each trajectory, compute KL(π_θ ‖ π_ref) ≈ mean_t(log π_θ - log π_ref).

    This is the per-token log-ratio averaged over all assistant tokens.
    A positive value means the policy has moved AWAY from the reference.
    Combined with λ_kl > 0 as a reward, this encourages anti-distillation:
    correct answers via trajectories that differ from the initial model.

    All forward passes use torch.no_grad() — gradients computed later
    inside GRPOTrainer.step().
    """
    kl_rewards = []
    for entry in group:
        raw = entry.get("log_data", {}).get("messages", [])
        cleaned = clean_messages_for_training(raw)
        if cleaned is None:
            kl_rewards.append(0.0)
            continue
        with torch.no_grad():
            lp_theta = compute_trajectory_log_probs(
                model, tokenizer, cleaned, device, max_length
            )
        if lp_theta is None:
            kl_rewards.append(0.0)
            continue
        mean_theta = lp_theta.mean().item()

        lp_ref = _ref_log_prob(model, tokenizer, cleaned, device, max_length)
        if lp_ref is None:
            kl_rewards.append(0.0)
        else:
            kl_rewards.append(mean_theta - lp_ref.item())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return kl_rewards


# ---------------------------------------------------------------------------
# Online rollout via vLLM subprocess (GPU 0,1, tp=2)
# ---------------------------------------------------------------------------

def _build_rollout_env() -> dict:
    """
    Subprocess env for vLLM rollout on GPU 0,1.

    - Sets CUDA_VISIBLE_DEVICES=0,1 so vLLM exclusively uses those 2 GPUs
      (training process runs on GPU 2,3 and is idle during rollout).
    - Strips distributed-training env vars so the subprocess sees no accelerate
      context and can initialise its own CUDA/NCCL cleanly.
    - HF_HUB_OFFLINE=1: avoid network calls during vLLM model loading.
    """
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_HOST_IP": "127.0.0.1",
        "CUDA_VISIBLE_DEVICES": "0",      # GPU 0 dedicated for vLLM (tp=1, 14B fits in 80G)
    }
    for k in [
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_TIMEOUT_KEEP_ALIVE",
        "NCCL_ASYNC_ERROR_HANDLING", "VLLM_USE_V1",
    ]:
        env.pop(k, None)
    return env


def rollout_batch(
    args,
    batch_questions: List[dict],
    ckpt_dir: str,
    step: int,
) -> Dict[str, List[dict]]:
    """
    Generate K trajectories per question in batch_questions using vLLM (tp=2).

    Creates a temp JSON with K copies of each question (K*B total entries).
    vLLM samples with temperature > 0, producing diverse trajectories for the
    same question.  Results are grouped by question text.

    Args:
        batch_questions: List of B question dicts (MATH-500 format).
        ckpt_dir:        Path to current LoRA checkpoint directory.
        step:            Training step number (used for output naming and seed).

    Returns:
        Dict mapping question_text → list of up to K scored trajectory dicts.
        Empty dict on failure.
    """
    # Duplicate each question K times → diverse trajectories via temperature
    entries = []
    for q in batch_questions:
        for _ in range(args.K):
            entries.append(dict(q))

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"grpo_rollout_step{step}_",
        dir="/tmp", delete=False,
    )
    json.dump({"metadata": {"n": len(entries)}, "examples": entries}, tmp)
    tmp.close()

    log_dir = Path(args.output_dir) / "rollouts" / f"step{step:05d}"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = _build_rollout_env()
    cmd = [
        sys.executable, "-m", "exps_research.rl_training.run_with_file_dist",
        "--experiment_type",    "agent",
        "--task_type",          "math",
        "--data_path",          tmp.name,
        "--model_type",         "vllm",
        "--model_id",           args.model_name,
        "--fine_tuned",
        "--lora_folder",        ckpt_dir,
        "--use_local_model",
        "--log_folder",         str(log_dir),
        "--n",                  "1",
        "--temperature",        str(args.rollout_temperature),
        "--top_p",              str(args.rollout_top_p),
        "--seed",               str(step % 10000),   # varies per step
        "--max_steps",          str(args.max_agent_steps),
        "--search_engine_type", "python_only",
        "--suffix",             f"online_step{step}",
        "--parallel_workers",   "1",     # offline vLLM is not thread-safe for multi-turn
    ]

    logger.info(
        f"[Rollout step={step}] {len(batch_questions)} questions × K={args.K} "
        f"= {len(entries)} trajectories via vLLM (GPU 0)"
    )
    t0 = datetime.now()
    # run_experiment.py saves outputs relative to lora_folder (ckpt_dir),
    # not to log_folder.  The temp JSON base name is used as a subdirectory:
    #   <ckpt_dir>/qa_results/<tmp_basename>*/evaluations/*_scored.jsonl
    tmp_basename = Path(tmp.name).stem   # e.g. "grpo_rollout_step7_abcd1234"
    result = subprocess.run(cmd, env=env, cwd=str(_ROOT))
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"[Rollout step={step}] done in {elapsed:.0f}s (rc={result.returncode})")

    try:
        os.unlink(tmp.name)
    except OSError:
        pass

    if result.returncode != 0:
        logger.warning(f"[Rollout step={step}] subprocess failed — skipping update")
        return {}

    # run_experiment.py writes to: <ckpt_dir>/qa_results/<tmp_basename>*/evaluations/*_scored.jsonl
    search_root = Path(ckpt_dir) / "qa_results"
    scored_files = list(search_root.glob(f"{tmp_basename}*/**/*_scored.jsonl"))
    if not scored_files:
        # Fallback: also check log_dir (in case run_experiment respected log_folder)
        scored_files = list(log_dir.glob("**/*_scored.jsonl"))
    logger.info(f"[Rollout step={step}] found {len(scored_files)} scored file(s)")

    # Group results by question text
    groups: Dict[str, List[dict]] = {}
    for scored_file in scored_files:
        if ".bak" in scored_file.name:
            continue
        with open(scored_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q_text = entry.get("question", "")
                groups.setdefault(q_text, []).append(entry)

    logger.info(
        f"[Rollout step={step}] collected {sum(len(v) for v in groups.values())} "
        f"trajectories across {len(groups)} questions"
    )
    return groups


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Disable NCCL watchdog: vLLM rollout (GPU 0,1) runs concurrently with
    # training ranks waiting at barrier.  Without this, the watchdog kills the
    # training process when rollout takes longer than the timeout.
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=12))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=None,
        kwargs_handlers=[pg_kwargs],
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
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        ds_cfg["train_micro_batch_size_per_gpu"] = 1
        ds_cfg["train_batch_size"] = (
            1 * args.gradient_accumulation_steps * accelerator.num_processes
        )

    model, optimizer = accelerator.prepare(model, optimizer)

    if torch.cuda.is_available() and is_main:
        alloc = torch.cuda.memory_allocated() / 1e9
        resv  = torch.cuda.memory_reserved()  / 1e9
        logger.info(f"GPU memory after model prepare: allocated={alloc:.1f}GB reserved={resv:.1f}GB")

    # Load question universe
    with open(args.question_json) as f:
        data = json.load(f)
    all_questions = data["examples"]
    random.shuffle(all_questions)
    logger.info(f"MATH-500 question set: {len(all_questions)} questions loaded")

    grpo = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        accelerator=accelerator,
        config={
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "kl_coeff": 0.0,   # KL folded into reward, not into GRPOTrainer loss
            "max_length": args.max_length,
            "clip_ratio": args.clip_ratio,
            "max_grad_norm": args.max_grad_norm,
        },
    )

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    global_step = args.initial_step
    optimizer_step = 0
    last_ckpt_step = global_step
    last_rollout_step = global_step - args.rollout_every   # trigger on first step

    # Current online pool: dict question_text → list of trajectories
    online_pool: Dict[str, List[dict]] = {}

    logger.info(
        f"Starting online GRPO (initial_step={global_step}, "
        f"rollout_every={args.rollout_every}, "
        f"checkpoint_every={args.checkpoint_every}, "
        f"B={args.batch_questions}, K={args.K}, "
        f"lambda_kl={args.lambda_kl})"
    )

    # ---- Initial checkpoint (needed before first vLLM rollout) ----
    if global_step == 0:
        initial_ckpt = save_checkpoint(model, tokenizer, args.output_dir, 0, accelerator)
    else:
        initial_ckpt = os.path.join(args.output_dir, f"checkpoint-step{global_step}")
    current_ckpt = initial_ckpt

    # ---- Question rotation pointer ----
    q_offset = 0

    while global_step < args.max_steps:
        # ---- Rollout: refresh online pool every rollout_every steps ----
        _rollout_due = (global_step - last_rollout_step) >= args.rollout_every
        if _rollout_due:
            last_rollout_step = global_step

            # Sample next batch of B questions (rotating window over all questions)
            batch_q = []
            for i in range(args.batch_questions):
                batch_q.append(all_questions[(q_offset + i) % len(all_questions)])
            q_offset = (q_offset + args.batch_questions) % len(all_questions)

            if is_main:
                new_pool = rollout_batch(args, batch_q, current_ckpt, global_step)
                # Merge into online_pool (or fully replace for freshest on-policy data)
                online_pool = new_pool

            # Broadcast pool availability (pool itself stays on main; other ranks skip)
            accelerator.wait_for_everyone()

            if not online_pool:
                logger.warning(f"[Step {global_step}] Rollout returned empty pool — skipping training step")
                global_step += 1
                continue

        if not online_pool:
            logger.warning(f"[Step {global_step}] No trajectories in pool yet — forcing rollout")
            last_rollout_step = global_step - args.rollout_every  # retrigger next iteration
            continue

        # ---- Training: sample one question-group from online_pool ----
        q_texts = list(online_pool.keys())
        if not q_texts:
            continue
        q_key = random.choice(q_texts)
        group = online_pool[q_key]
        if len(group) < 2:
            # Need at least 2 trajectories for group-norm advantage
            continue

        # Subsample to n_trajs_per_question if pool is large
        if len(group) > args.n_trajs_per_question:
            group = random.sample(group, args.n_trajs_per_question)

        # ---- Compute rewards (all processes participate for ZeRO-3 forward) ----
        r_task = compute_r_task(group)
        r_kl   = compute_kl_rewards(
            model, tokenizer, group, accelerator.device, args.max_length
        )
        r_total = [
            rt + args.lambda_kl * rk
            for rt, rk in zip(r_task, r_kl)
        ]

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
            n = max(len(r_task), 1)
            log_entry = {
                "step": global_step,
                "optimizer_step": optimizer_step,
                "loss": info.get("loss", 0.0),
                "r_task_mean": sum(r_task) / n,
                "r_kl_mean":  sum(r_kl)   / n,
                "r_total_mean": sum(r_total) / n,
                "pool_questions": len(online_pool),
                "group_size": len(group),
            }
            logger.info(json.dumps(log_entry))
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # ---- Checkpoint ----
        _ckpt_due = (
            args.checkpoint_every > 0
            and global_step % args.checkpoint_every == 0
            and global_step > last_ckpt_step
        )
        _rollout_due_next = (global_step - last_rollout_step) >= args.rollout_every
        if _ckpt_due or _rollout_due_next:
            if global_step > last_ckpt_step:
                last_ckpt_step = global_step
                current_ckpt = save_checkpoint(
                    model, tokenizer, args.output_dir, global_step, accelerator
                )

    # Final checkpoint
    if global_step > last_ckpt_step:
        save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
    if is_main:
        logger.info("Online GRPO training complete.")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Online GRPO on MATH-500 with Qwen3-14B")

    # Data
    p.add_argument("--question_json",  type=str, required=True,
                   help="MATH-500 question JSON (same format as pilot_question_json)")

    # Model
    p.add_argument("--model_name",    type=str, default="Qwen/Qwen3-14B")
    p.add_argument("--lora_r",        type=int, default=16,
                   help="LoRA rank (alpha = 2r)")
    p.add_argument("--max_length",    type=int, default=4096,
                   help="Max token length for log-prob computation")

    # Rollout
    p.add_argument("--K",             type=int, default=8,
                   help="Trajectories to collect per question per rollout")
    p.add_argument("--batch_questions", type=int, default=4,
                   help="Questions per rollout batch")
    p.add_argument("--rollout_every", type=int, default=8,
                   help="Refresh trajectory pool every N training steps")
    p.add_argument("--rollout_temperature", type=float, default=0.8)
    p.add_argument("--rollout_top_p",       type=float, default=0.9)
    p.add_argument("--max_agent_steps",     type=int,   default=5,
                   help="Max CodeAct steps per trajectory")
    p.add_argument("--n_trajs_per_question", type=int, default=8,
                   help="Max trajectories sampled per question for one GRPO step")

    # Reward
    p.add_argument("--lambda_kl",     type=float, default=0.1,
                   help="Weight for KL(π_θ ‖ π_ref) divergence reward. "
                        "Positive = reward trajectories that differ from base model.")

    # Training
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--max_steps",     type=int,   default=400)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--clip_ratio",    type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)

    # Checkpointing
    p.add_argument("--checkpoint_every", type=int, default=50,
                   help="Save LoRA checkpoint every N training steps")

    # Resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--initial_step",           type=int, default=0)

    # Output / logging
    p.add_argument("--output_dir",  type=str,
                   default="training_outputs/qwen3-14B/grpo_online")
    p.add_argument("--log_every",   type=int, default=5)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
