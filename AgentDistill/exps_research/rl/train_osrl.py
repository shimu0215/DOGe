"""
OS-RL: Output-Sensitivity RL training script.

Training loop (one RL iteration):
  1. Save current LoRA adapter → disk
  2. Start vLLM subprocess serving base + LoRA adapter
  3. Collect G agent rollouts per problem (batch_size problems)
  4. Kill vLLM; free GPU memory
  5. Load training model (FSDP/LoRA); compute R_task + R_sensitivity
  6. GRPO policy-gradient update (with KL penalty vs reference model)
  7. Save checkpoint; repeat

Launch with torchrun:
    torchrun --nproc_per_node=4 -m exps_research.rl.train_osrl [args]

Or via the launch script:
    bash scripts/training/train_osrl_math_32b.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from exps_research.rl.osrl_config import OSRLConfig
from exps_research.rl.osrl_reward import compute_rewards

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("osrl")


# =========================================================================== #
# Utility helpers
# =========================================================================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str, level: str = "info") -> None:
    if is_main_process():
        getattr(logger, level)(msg)


# =========================================================================== #
# Data loading
# =========================================================================== #

def load_math_problems(data_path: str) -> List[Dict]:
    with open(data_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    return data


# =========================================================================== #
# vLLM subprocess management
# =========================================================================== #

def save_lora_for_vllm(model, output_path: str) -> None:
    """Save the LoRA adapter weights so vLLM can load them."""
    if is_main_process():
        os.makedirs(output_path, exist_ok=True)
        # Unwrap FSDP to access the LoRA model
        unwrapped = model
        if isinstance(model, FSDP):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = model.state_dict()
            unwrapped = model.module
        else:
            state_dict = model.state_dict()
        unwrapped.save_pretrained(output_path, state_dict=state_dict)
    if dist.is_initialized():
        dist.barrier()


def start_vllm_server(config: OSRLConfig, lora_path: str, serve_log: str) -> subprocess.Popen:
    """Launch vLLM as a subprocess; return the Popen object."""
    cmd = [
        sys.executable, "serve_vllm.py",
        "--model", config.model_name,
        "--tensor-parallel-size", str(config.vllm_tp_size),
        "--port", str(config.vllm_port),
        "--gpu-memory-utilization", str(config.vllm_gpu_util),
        "--disable-log-requests",
        "--disable-log-stats",
        "--lora-modules", f"osrl_policy={lora_path}",
        "--max-lora-rank", str(config.lora_rank),
    ]
    log_file = open(serve_log, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    return proc


def wait_for_vllm(serve_log: str, timeout: int = 1800) -> bool:
    waited = 0
    while waited < timeout:
        if os.path.exists(serve_log):
            with open(serve_log) as f:
                if "Application startup complete." in f.read():
                    return True
        time.sleep(5)
        waited += 5
    return False


def kill_vllm(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        proc.kill()


# =========================================================================== #
# Rollout collection
# =========================================================================== #

def collect_rollouts(
    config: OSRLConfig,
    problems: List[Dict],
    lora_path: str,
    rollout_dir: str,
    iteration: int,
    rollout_seed: int,
) -> List[Dict]:
    """
    Collect G rollouts per problem by calling the existing collect script.
    Returns flat list of trajectory dicts (len = len(problems) * G, with errors possible).
    """
    all_trajectories: List[Dict] = []

    for g in range(config.num_rollouts_per_problem):
        seed = rollout_seed + g * 1000 + iteration
        out_dir = os.path.join(rollout_dir, f"iter{iteration:04d}_g{g}")
        os.makedirs(out_dir, exist_ok=True)

        # Write the problem batch as a temporary JSON file
        batch_path = os.path.join(out_dir, "problems.json")
        with open(batch_path, "w") as f:
            json.dump(problems, f)

        # Call the existing run_experiment entry point
        cmd = [
            sys.executable, "-m",
            "exps_research.unified_framework.run_experiment",
            "--experiment_type", "agent",
            "--data_path", batch_path,
            "--model_type", "vllm",
            "--model_id", config.model_name,
            "--log_folder", out_dir,
            "--max_tokens", str(config.max_tokens),
            "--multithreading",
            "--use_process_pool",
            "--parallel_workers", str(config.parallel_workers),
            "--n", "1",
            "--temperature", str(config.rollout_temperature),
            "--top_p", str(config.rollout_top_p),
            "--seed", str(seed),
            "--max_steps", str(config.max_agent_steps),
            "--search_engine_type", "python_only",
            "--use_single_endpoint",
            "--suffix", f"osrl_g{g}",
            "--fine_tuned",
            "--lora_folder", lora_path,
        ]
        log(f"[rollout] iter={iteration} g={g} seed={seed}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"[rollout] WARNING: run_experiment returned {result.returncode}", "warning")
            log(result.stderr[-2000:], "warning")

        # Read results
        dataset_name = Path(config.data_path).stem
        model_name = Path(config.model_name).name
        jsonl_path = os.path.join(
            out_dir,
            f"{dataset_name}_test",
            f"{model_name}_temp=0.7_n=1_seed={seed}_type=agent_steps="
            f"{config.max_agent_steps}_python_only_osrl_g{g}.jsonl",
        )
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_trajectories.append(json.loads(line))
        else:
            log(f"[rollout] WARNING: expected output not found at {jsonl_path}", "warning")

    return all_trajectories


# =========================================================================== #
# Trajectory tokenisation for GRPO loss
# =========================================================================== #

_ASSISTANT_ROLES = {"assistant", "tool-call"}
_MASKED_LABEL = -100


def _get_text(content) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def tokenise_trajectory(
    trajectory: Dict,
    tokenizer,
    max_length: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert a trajectory into (input_ids, labels) tensors.

    labels[i] = token_id  for tokens in assistant / tool-call turns
    labels[i] = -100      for everything else (system, user, tool-response)

    Returns None if the trajectory has no log_data or exceeds max_length.
    """
    messages = (trajectory.get("log_data") or {}).get("messages", [])
    if not messages:
        return None

    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool-call": "assistant",
        "tool-response": "user",
    }

    # Build flattened messages with role labels preserved for masking
    flat_messages = []
    is_action = []   # parallel list: True if this msg contributes to the loss
    for msg in messages:
        role = role_map.get(msg["role"], "user")
        text = _get_text(msg["content"])
        if flat_messages and flat_messages[-1]["role"] == role:
            flat_messages[-1]["content"] += "\n" + text
            is_action[-1] = is_action[-1] or (msg["role"] in _ASSISTANT_ROLES)
        else:
            flat_messages.append({"role": role, "content": text})
            is_action.append(msg["role"] in _ASSISTANT_ROLES)

    # Tokenise full conversation
    full_ids = tokenizer.apply_chat_template(
        flat_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )[0]  # (L,)

    if full_ids.shape[0] > max_length:
        return None

    # Build labels by re-tokenising each message and aligning spans
    labels = torch.full_like(full_ids, _MASKED_LABEL)
    cursor = 0
    for i, msg in enumerate(flat_messages):
        # Tokenise prefix up to and including this message
        prefix_ids = tokenizer.apply_chat_template(
            flat_messages[: i + 1],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )[0]
        end = prefix_ids.shape[0]
        if is_action[i]:
            labels[cursor:end] = full_ids[cursor:end]
        cursor = end

    return full_ids, labels


# =========================================================================== #
# GRPO loss computation
# =========================================================================== #

def compute_grpo_loss(
    model,
    ref_model,
    tokenizer,
    trajectories: List[Dict],
    rewards: List[float],
    config: OSRLConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    GRPO loss for one batch of trajectories.

    Trajectories are assumed to be grouped by question:
        trajectories[i*G : (i+1)*G] are the G rollouts for question i.
    rewards is a parallel list.
    """
    G = config.num_rollouts_per_problem
    n_questions = len(trajectories) // G

    total_pg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_kl_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    model.train()

    for q in range(n_questions):
        group = trajectories[q * G : (q + 1) * G]
        group_rewards = rewards[q * G : (q + 1) * G]

        # Group-relative advantage normalisation
        r_arr = np.array(group_rewards, dtype=np.float32)
        mean_r = r_arr.mean()
        std_r = r_arr.std() + config.grpo_eps
        advantages = (r_arr - mean_r) / std_r

        for traj, adv in zip(group, advantages):
            tok = tokenise_trajectory(traj, tokenizer, config.max_seq_length)
            if tok is None:
                continue
            input_ids, labels = tok
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            # --- current policy log probs ---
            logits = model(input_ids=input_ids).logits          # (1, L, V)
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            action_mask = shifted_labels != _MASKED_LABEL

            if not action_mask.any():
                continue

            log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
            token_lps = log_probs.gather(
                -1, shifted_labels.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)                                        # (1, L-1)
            action_lps = token_lps[action_mask]                  # (N_action,)
            seq_log_prob = action_lps.sum()

            # --- reference log probs ---
            with torch.no_grad():
                ref_logits = ref_model(input_ids=input_ids).logits
                ref_shifted = ref_logits[:, :-1, :]
                ref_log_probs = F.log_softmax(ref_shifted.float(), dim=-1)
                ref_token_lps = ref_log_probs.gather(
                    -1, shifted_labels.clamp(min=0).unsqueeze(-1)
                ).squeeze(-1)
                ref_action_lps = ref_token_lps[action_mask]
                ref_seq_log_prob = ref_action_lps.sum()

            # Policy gradient (negative because we maximise reward)
            pg = -float(adv) * seq_log_prob
            # KL divergence: KL(current || ref) = Σ p * (log p - log ref)
            kl = (action_lps - ref_action_lps).sum()

            total_pg_loss = total_pg_loss + pg
            total_kl_loss = total_kl_loss + kl
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = (total_pg_loss + config.kl_coef * total_kl_loss) / n_valid
    return loss


# =========================================================================== #
# Model initialisation
# =========================================================================== #

def build_model_and_tokenizer(config: OSRLConfig, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading base model {config.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Attach LoRA
    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base_model, lora_cfg)
    policy.print_trainable_parameters()

    # Reference model: same base, no LoRA, frozen
    log("Loading reference model ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return policy, ref_model, tokenizer


# =========================================================================== #
# Main training loop
# =========================================================================== #

def train(config: OSRLConfig) -> None:
    # ---- distributed setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    set_seed(config.seed + local_rank)

    os.makedirs(config.output_dir, exist_ok=True)
    rollout_root = os.path.join(config.output_dir, "rollouts")
    ckpt_root = os.path.join(config.output_dir, "checkpoints")
    lora_export_dir = os.path.join(config.output_dir, "lora_for_vllm")
    serve_log = os.path.join(config.output_dir, "vllm_serve.log")

    # ---- load problems ----
    problems = load_math_problems(config.data_path)
    log(f"Loaded {len(problems)} problems from {config.data_path}")

    # ---- build models ----
    policy, ref_model, tokenizer = build_model_and_tokenizer(config, device)

    # Wrap policy with FSDP for multi-GPU training
    if dist.is_initialized() and dist.get_world_size() > 1:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        auto_wrap = transformer_auto_wrap_policy(
            transformer_layer_cls={Qwen3DecoderLayer}
        )
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        policy = FSDP(
            policy.to(device),
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device,
        )
    else:
        policy = policy.to(device)

    # ---- optimiser & scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )
    total_steps = config.num_rl_iterations * config.grpo_epochs_per_iter
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # ---- resume ----
    start_iter = 0
    if config.resume_from_checkpoint and os.path.isdir(config.resume_from_checkpoint):
        log(f"Resuming from {config.resume_from_checkpoint}")
        # TODO: load optimizer/scheduler state
        start_iter = int(Path(config.resume_from_checkpoint).name.split("iter")[-1])

    # ---- main RL loop ----
    for iteration in range(start_iter, config.num_rl_iterations):
        log(f"\n{'='*60}")
        log(f"RL iteration {iteration + 1} / {config.num_rl_iterations}")

        # Sample a batch of problems
        batch_problems = random.sample(
            problems,
            min(config.rollout_batch_size, len(problems)),
        )

        # ---------------------------------------------------------------- #
        # Phase 1: rollout collection via vLLM
        # ---------------------------------------------------------------- #
        if is_main_process():
            # Export current LoRA weights for vLLM to load
            save_lora_for_vllm(policy, lora_export_dir)
            vllm_proc = start_vllm_server(config, lora_export_dir, serve_log)
            log("Waiting for vLLM server ...")
            ok = wait_for_vllm(serve_log, timeout=config.vllm_startup_timeout)
            if not ok:
                log("vLLM failed to start; skipping this iteration.", "warning")
                kill_vllm(vllm_proc)
                continue

            trajectories = collect_rollouts(
                config, batch_problems, lora_export_dir,
                rollout_root, iteration, config.rollout_seed,
            )
            log(f"Collected {len(trajectories)} trajectories")
            kill_vllm(vllm_proc)

            # ------------------------------------------------------------ #
            # Phase 2: reward computation (sensitivity uses training model)
            # ------------------------------------------------------------ #
            rewards, r_task, r_sens = compute_rewards(
                policy, tokenizer, trajectories, device,
                lambda_sensitivity=config.lambda_sensitivity,
                mask_placeholder=config.sensitivity_mask_token,
                sensitivity_max_steps=config.sensitivity_max_steps,
            )
            mean_r = np.mean(rewards) if rewards else 0.0
            mean_rt = np.mean(r_task) if r_task else 0.0
            mean_rs = np.mean(r_sens) if r_sens else 0.0
            log(
                f"Rewards  total={mean_r:.4f}  task={mean_rt:.4f}"
                f"  sensitivity={mean_rs:.4f} (λ={config.lambda_sensitivity})"
            )
        else:
            trajectories, rewards = [], []

        # Broadcast trajectories/rewards to all ranks (simple: only rank 0 trains)
        # For multi-rank training, broadcast via file or dist collective.
        # Current implementation: all ranks participate in GRPO loss.
        if dist.is_initialized():
            dist.barrier()

        # ---------------------------------------------------------------- #
        # Phase 3: GRPO update
        # ---------------------------------------------------------------- #
        for grpo_epoch in range(config.grpo_epochs_per_iter):
            optimizer.zero_grad()

            loss = compute_grpo_loss(
                policy, ref_model, tokenizer,
                trajectories, rewards, config, device,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            log(
                f"  [grpo_epoch {grpo_epoch+1}] loss={loss.item():.6f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # ---------------------------------------------------------------- #
        # Checkpoint
        # ---------------------------------------------------------------- #
        if is_main_process() and (iteration + 1) % config.save_every_n_iters == 0:
            ckpt_path = os.path.join(ckpt_root, f"iter{iteration + 1:04d}")
            save_lora_for_vllm(policy, ckpt_path)
            log(f"Saved checkpoint to {ckpt_path}")

    # Final save
    if is_main_process():
        final_path = os.path.join(ckpt_root, "final")
        save_lora_for_vllm(policy, final_path)
        log(f"Training complete. Final checkpoint: {final_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


# =========================================================================== #
# CLI entry point
# =========================================================================== #

def parse_args() -> OSRLConfig:
    parser = argparse.ArgumentParser(description="OS-RL training")

    # Required
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-32B")

    # Optional overrides (mirror OSRLConfig fields)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lambda_sensitivity", type=float, default=0.1)
    parser.add_argument("--num_rollouts_per_problem", type=int, default=4)
    parser.add_argument("--rollout_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_rl_iterations", type=int, default=100)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vllm_tp_size", type=int, default=4)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--max_seq_length", type=int, default=10240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_n_iters", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", default=None)

    args = parser.parse_args()
    cfg = OSRLConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    config = parse_args()
    train(config)
