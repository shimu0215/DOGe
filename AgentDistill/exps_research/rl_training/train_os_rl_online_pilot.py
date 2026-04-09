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
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
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
                # vLLM's LoRA loader expects keys like:
                #   base_model.model.layers.X.self_attn.q_proj.lora_A.weight
                # but PEFT named_parameters() returns:
                #   base_model.model.layers.X.self_attn.q_proj.lora_A.default.weight
                # Strip the adapter name ("default") so vLLM can load the weights.
                import re as _re
                def _clean_lora_key(name: str) -> str:
                    return _re.sub(
                        r'\.(lora_A|lora_B|lora_embedding_A|lora_embedding_B)'
                        r'\.([^.]+)\.',
                        r'.\1.',
                        name,
                    )
                param_dict = {
                    _clean_lora_key(name): param.data.detach().cpu().clone()
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


def resample_trajectories(args, ckpt_dir: str, step: int) -> List[str]:
    """
    Collect fresh trajectories using the current LoRA checkpoint via vLLM offline.

    Uses run_experiment --model_type vllm --use_local_model --fine_tuned with
    local_device_id=3 (GPU 3, ~79 GB free alongside ZeRO-3 ~1 GB shards).
    This calls VLLMModel which uses vllm.LLM (offline V0, no HTTP server, no
    EngineCore subprocess) — avoiding the vLLM V1 IPC/TCPStore SLURM issue.
    """
    dataset_name = Path(args.pilot_question_json).stem   # e.g. "pilot_questions"
    model_name = Path(args.model_name).name              # e.g. "Qwen3-32B"
    log_root = Path(ckpt_dir) / "qa_results"

    # Build env: strip all distributed training vars so the subprocess starts clean.
    # HF_HUB_OFFLINE=1: prevent vLLM from querying the HuggingFace API at init time;
    #   the model is already fully cached under $HF_HOME.
    # VLLM_HOST_IP=127.0.0.1: vLLM V1 LLM spawns EngineCore as a subprocess and
    #   the parent creates a TCPStore server at get_ip():port.  In SLURM, get_ip()
    #   returns the node's external IP (172.16.x.x) which is unreachable from within
    #   the cgroup.  Setting VLLM_HOST_IP overrides get_ip() to 127.0.0.1 so both
    #   parent and child communicate over loopback.
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # VLLM_HOST_IP=127.0.0.1 alone is insufficient: SLURM's network
        # namespace isolation blocks TCP connections even on loopback.
        # VLLM_ENABLE_V1_MULTIPROCESSING=0 makes vLLM use SyncInProcClient
        # instead of spawning a separate EngineCore subprocess, removing all
        # TCPStore / network dependencies entirely.
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
               "MASTER_ADDR", "MASTER_PORT",
               "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
               "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_TIMEOUT_KEEP_ALIVE",
               "NCCL_ASYNC_ERROR_HANDLING",
               "VLLM_USE_V1"]:
        env.pop(_k, None)

    new_files = []
    for seed in args.resample_seeds:
        logger.info(f"Resampling seed={seed} from checkpoint: {ckpt_dir}")
        cmd = [
            # Use run_with_file_dist wrapper: patches torch.distributed.init_process_group
            # to replace tcp:// with file:// (FileStore) before vLLM initializes.
            # This avoids the TCPStore 600s timeout that occurs when UniProcExecutor
            # calls init_distributed_environment inside the SLURM training job.
            sys.executable, "-m", "exps_research.rl_training.run_with_file_dist",
            "--experiment_type",    "agent",
            "--task_type",          "math",
            "--data_path",          args.pilot_question_json,
            "--model_type",         "vllm",
            "--model_id",           args.model_name,
            "--fine_tuned",
            "--lora_folder",        ckpt_dir,
            "--use_local_model",
            "--log_folder",         str(log_root),
            "--n",                  "1",
            "--temperature",        "0.7",
            "--top_p",              "0.8",
            "--seed",               str(seed),
            "--max_steps",          str(args.max_agent_steps),
            "--search_engine_type", "python_only",
            "--suffix",             f"python_only_seed{seed}",
            "--parallel_workers",   "1",   # offline LLM is not thread-safe
            "--max_model_len",      "16384",  # limit KV cache usage: 32B model uses ~64 GB, leaving ~7-8 GB for KV cache; 40960 (default) requires 10 GB but only 7.71 GB is available
        ]
        result = subprocess.run(cmd, env=env, cwd=str(_ROOT))
        if result.returncode != 0:
            logger.warning(f"run_experiment failed seed={seed} (rc={result.returncode})")
            continue

        # Locate scored output file (_scored.jsonl has the 'score' field for R_task)
        found = list(log_root.glob(
            f"**/*seed={seed}*python_only_seed{seed}*_scored.jsonl"
        ))
        # Exclude backup files
        found = [p for p in found if ".bak" not in p.name]
        new_files.extend([str(p) for p in found])
        if found:
            logger.info(f"Collected: {found}")
        else:
            logger.warning(f"No output file found for seed={seed} in {log_root}")

    logger.info(f"Resampling produced {len(new_files)} new files at step {step}.")
    return new_files


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Extend NCCL process-group timeout to 3 hours so that the vLLM resampling
    # step (rank 0 only, can take ~30 min per seed × 3 seeds) does not trigger
    # the 30-min watchdog on ranks 1-3 that are waiting at the barrier.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))
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
            # NOTE: all ranks must call save_checkpoint (uses GatheredParameters + barriers).
            _ckpt_due = (
                args.resample_every > 0
                and (global_step == 1 or global_step % args.resample_every == 0)
                and global_step > last_resample_step
            )
            if _ckpt_due:
                last_resample_step = global_step
                ckpt_dir = save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
                if is_main:
                    new_files = resample_trajectories(args, ckpt_dir, global_step)
                    if new_files:
                        # add_files merges new trajectories into the existing pool
                        # instead of replacing it. This is important when n=1 per
                        # question: with only 1 new trajectory per question the pool
                        # would be empty (min_group_size=2). By accumulating new
                        # trajectories on top of the initial offline set, each question
                        # keeps enough trajectories to stay valid for GRPO.
                        pool.add_files(new_files)
                        logger.info(f"Pool updated with {len(new_files)} new files. "
                                    f"Stats: {pool.stats()}")
                # All ranks wait for rank 0 to finish resampling before continuing
                # training; without this barrier the other ranks proceed to the next
                # optimizer.step() while rank 0 is still inside vLLM, causing a 30-min
                # NCCL watchdog timeout on the pending all-reduce collective.
                accelerator.wait_for_everyone()

        if global_step >= args.max_steps:
            break

    if global_step > last_resample_step:
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
