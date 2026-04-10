"""
train_os_rl_online_pilot.py — Semi-online OS-RL (full-scale, 500 questions).

Changes vs. pilot version:
  - Replace strategy: new trajectories REPLACE old ones (FIFO per question),
    keeping pool size stable and shifting distribution toward on-policy data.
  - Partial resampling: only `n_resample_questions` questions updated per cycle
    (~10% of total), controlled by rotating random selection to cover all
    questions across resamples.
  - Pre-training data quality check: sample 50 questions, require ≥30% accuracy.
  - Separate checkpoint_every from resample_every: checkpoints saved more
    frequently than resampling (which involves expensive vLLM inference).
  - First resample fires at resample_every (not at step 1) for full-scale run.

GPU layout during resampling:
  Training  : 4 processes on GPUs 0-3 (ZeRO-3 CPU offload, ~1 GB each)
  vLLM      : tp-size=1, one GPU (uses the ~79 GB free on that GPU)
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
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta, datetime
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

def glob_scored_files(trajectory_dir: str, seed_list: Optional[List[int]] = None) -> List[str]:
    """Glob pre-scored JSONL files from trajectory_dir.

    If seed_list is given, filter by seed; otherwise return all scored files.
    Also accepts direct .jsonl file paths (for the full-scale pool that stores
    all seeds as separate files in arbitrary locations).
    """
    d = Path(trajectory_dir)
    if not d.exists():
        logger.warning(f"trajectory_dir not found: {trajectory_dir}")
        return []

    files = []
    if seed_list:
        for seed in seed_list:
            matches = sorted(d.glob(f"*seed={seed}_*_scored.jsonl"))
            files.extend([str(m) for m in matches])
    if not files:
        # Fall back: all scored JSONL files in the directory (non-recursive)
        files = [str(p) for p in sorted(d.glob("*_scored.jsonl"))]
    if not files:
        # Try recursive (files may be in subdirs)
        files = [str(p) for p in sorted(d.rglob("*_scored.jsonl"))]
    logger.info(f"Found {len(files)} trajectory files in {trajectory_dir}.")
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


def check_data_quality(
    pool: TrajectoryPool,
    n_sample: int = 50,
    min_accuracy: float = 0.30,
) -> bool:
    """Sample n_sample questions from pool and verify accuracy >= min_accuracy.

    Raises RuntimeError if accuracy is below threshold.
    Returns True on success.
    """
    qs = pool.valid_questions
    if not qs:
        raise RuntimeError("Pool is empty — no valid questions found.")
    sampled = random.sample(qs, min(n_sample, len(qs)))
    total = correct = 0
    for q in sampled:
        for entry in pool.pool[q]:
            total += 1
            if entry.get("score", False):
                correct += 1
    acc = correct / total if total > 0 else 0.0
    logger.info(
        f"[Quality check] Sampled {len(sampled)} questions, "
        f"{correct}/{total} correct, accuracy={acc:.1%} "
        f"(threshold={min_accuracy:.1%})"
    )
    if acc < min_accuracy:
        raise RuntimeError(
            f"Data quality check FAILED: accuracy {acc:.1%} < {min_accuracy:.1%}. "
            "Check the trajectory files or lower --quality_min_acc."
        )
    logger.info("[Quality check] PASSED.")
    return True


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


def resample_trajectories(
    args,
    ckpt_dir: str,
    step: int,
    seed: int,
    questions_subset: Optional[List[dict]] = None,
) -> List[str]:
    """
    Collect fresh trajectories using the current LoRA checkpoint via vLLM offline.

    Args:
        questions_subset: If given, only resample these questions (list of dicts
                          with keys: id, question, answer, ...).
                          If None, resample all pilot questions.
        seed: Random seed for this resampling run.

    Returns list of scored JSONL file paths.
    """
    log_root = Path(ckpt_dir) / "qa_results"
    log_root.mkdir(parents=True, exist_ok=True)

    # Build env: strip all distributed training vars so the subprocess starts clean.
    # Also reset CUDA_VISIBLE_DEVICES: accelerate sets it to the single GPU for
    # this rank (e.g. "0"), but vLLM tp>1 needs to see ALL GPUs.
    # Use WORLD_SIZE (= number of GPUs) read BEFORE popping distributed vars.
    _world_size = int(os.environ.get("WORLD_SIZE", 1))
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_HOST_IP": "127.0.0.1",
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(_world_size)),
    }
    for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
               "MASTER_ADDR", "MASTER_PORT",
               "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
               "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_TIMEOUT_KEEP_ALIVE",
               "NCCL_ASYNC_ERROR_HANDLING",
               "VLLM_USE_V1"]:
        env.pop(_k, None)

    # Write a temporary JSON file with the subset of questions to resample
    if questions_subset is not None:
        tmp_json = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"resample_step{step}_",
            dir="/tmp", delete=False
        )
        json.dump({"metadata": {"n": len(questions_subset)},
                   "examples": questions_subset}, tmp_json)
        tmp_json.close()
        data_path = tmp_json.name
        logger.info(f"Resampling {len(questions_subset)} questions (seed={seed}) "
                    f"from checkpoint: {ckpt_dir}")
    else:
        data_path = args.pilot_question_json
        tmp_json = None
        logger.info(f"Resampling ALL questions (seed={seed}) from checkpoint: {ckpt_dir}")

    try:
        cmd = [
            sys.executable, "-m", "exps_research.rl_training.run_with_file_dist",
            "--experiment_type",    "agent",
            "--task_type",          "math",
            "--data_path",          data_path,
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
            "--max_model_len",      "24576",
            "--tensor_parallel_size", str(args.vllm_tp_size),
        ]
        result = subprocess.run(cmd, env=env, cwd=str(_ROOT))
        if result.returncode != 0:
            logger.warning(f"run_experiment failed seed={seed} (rc={result.returncode})")
            return []
    finally:
        if tmp_json is not None:
            try:
                os.unlink(tmp_json.name)
            except OSError:
                pass

    # Locate scored output file
    found = list(log_root.glob(f"**/*seed={seed}*python_only_seed{seed}*_scored.jsonl"))
    found = [p for p in found if ".bak" not in p.name]
    new_files = [str(p) for p in found]
    if new_files:
        logger.info(f"Step {step}: collected {len(new_files)} new file(s): {new_files}")
    else:
        logger.warning(f"Step {step}: no output file found for seed={seed} in {log_root}")

    return new_files


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Disable NCCL watchdog so that vLLM resampling on rank0 (which can take
    # 1-3 hours loading 32B from NFS) does not kill ranks 1-3 waiting at the
    # barrier. TORCH_NCCL_ENABLE_MONITORING=0 is set in the launch script;
    # this large timeout is a belt-and-suspenders fallback.
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=24))
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

    # Load initial pool from pre-collected offline data
    # trajectory_dir may contain files directly (for full-scale, files are
    # copied from various sources into a single flat directory).
    traj_files = glob_scored_files(args.trajectory_dir)
    pool = TrajectoryPool(traj_files, min_group_size=args.min_group_size)
    if is_main:
        logger.info(f"Initial pool: {pool.stats()}")

    # --- Data quality check (main process only, but block all until done) ---
    if is_main:
        check_data_quality(pool, n_sample=50, min_accuracy=args.quality_min_acc)
    accelerator.wait_for_everyone()

    # Load pilot questions for resampling
    with open(args.pilot_question_json) as f:
        pilot_data = json.load(f)
    all_pilot_questions = pilot_data["examples"]   # list of dicts
    logger.info(f"Pilot question set: {len(all_pilot_questions)} questions "
                f"(will resample {args.n_resample_questions} per cycle).")

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
    last_ckpt_step = global_step
    resample_cycle = 0   # counts how many resamples have been done

    logger.info(f"Starting semi-online OS-RL (initial_step={global_step}, "
                f"resample_every={args.resample_every}, "
                f"checkpoint_every={args.checkpoint_every}, "
                f"n_resample_questions={args.n_resample_questions}, "
                f"seeds_per_resample={args.seeds_per_resample})...")

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

            # ---- Checkpoint (frequent) ----
            # NOTE: all ranks must call save_checkpoint (uses GatheredParameters + barriers).
            _ckpt_due = (
                args.checkpoint_every > 0
                and global_step % args.checkpoint_every == 0
                and global_step > last_ckpt_step
            )
            if _ckpt_due:
                last_ckpt_step = global_step
                save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)

            # ---- Resample (less frequent, uses vLLM) ----
            _resample_due = (
                args.resample_every > 0
                and (global_step == 1 or global_step % args.resample_every == 0)
                and global_step > last_resample_step
            )
            if _resample_due:
                last_resample_step = global_step

                # Save checkpoint before resampling if not already saved at this step
                if global_step != last_ckpt_step:
                    last_ckpt_step = global_step
                    ckpt_dir = save_checkpoint(
                        model, tokenizer, args.output_dir, global_step, accelerator
                    )
                else:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-step{global_step}")

                if is_main:
                    _resample_start = datetime.now()
                    logger.info(
                        f"[RESAMPLE START] step={global_step} "
                        f"time={_resample_start.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    # Each seed in this cycle gets its OWN non-overlapping
                    # question subset of size n_resample_questions.
                    # Together seeds_per_resample seeds cover
                    # seeds_per_resample * n_resample_questions questions per cycle.
                    # The window rotates across cycles for full coverage over time.
                    n_q   = args.n_resample_questions   # questions per seed
                    n_spr = args.seeds_per_resample      # seeds per cycle
                    n_total = len(all_pilot_questions)

                    # Starting offset for this cycle (rotate across cycles)
                    cycle_offset = (resample_cycle * n_spr * n_q) % n_total

                    # Select seeds for this cycle (rotate through resample_seeds pool)
                    seed_start = (resample_cycle * n_spr) % len(args.resample_seeds)
                    seeds_this_cycle = [
                        args.resample_seeds[(seed_start + i) % len(args.resample_seeds)]
                        for i in range(n_spr)
                    ]
                    resample_cycle += 1

                    logger.info(
                        f"Resampling cycle {resample_cycle}: "
                        f"{n_spr} seeds × {n_q} questions = {n_spr * n_q} total, "
                        f"seeds={seeds_this_cycle}, cycle_offset={cycle_offset}"
                    )
                    all_new_files = []
                    for i, resample_seed in enumerate(seeds_this_cycle):
                        # Non-overlapping slice for this seed
                        start = (cycle_offset + i * n_q) % n_total
                        if start + n_q <= n_total:
                            subset = all_pilot_questions[start : start + n_q]
                        else:
                            subset = (all_pilot_questions[start:]
                                      + all_pilot_questions[: (start + n_q) % n_total])
                        logger.info(
                            f"  seed={resample_seed}: questions [{start}, {start+n_q}) "
                            f"(wraps={start + n_q > n_total})"
                        )
                        new_files = resample_trajectories(
                            args, ckpt_dir, global_step,
                            seed=resample_seed,
                            questions_subset=subset,
                        )
                        all_new_files.extend(new_files)

                    if all_new_files:
                        # Replace strategy: for each new trajectory, remove one
                        # old trajectory from that question's pool (FIFO).
                        n_replaced = pool.replace_with_files(all_new_files)
                        logger.info(
                            f"Pool updated: replaced {n_replaced} trajectories "
                            f"({len(all_new_files)} files). Stats: {pool.stats()}"
                        )
                    else:
                        logger.warning("Resampling produced no new files — pool unchanged.")

                    _resample_end = datetime.now()
                    _resample_elapsed = (_resample_end - _resample_start).total_seconds()
                    logger.info(
                        f"[RESAMPLE END] step={global_step} "
                        f"time={_resample_end.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"elapsed={_resample_elapsed:.0f}s "
                        f"({_resample_elapsed/60:.1f}min)"
                    )

                # All ranks wait for rank 0 to finish resampling
                accelerator.wait_for_everyone()

        if global_step >= args.max_steps:
            break

    # Final checkpoint
    if global_step > last_ckpt_step:
        save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
    if is_main:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Semi-online OS-RL (full-scale)")

    # Data
    p.add_argument("--trajectory_dir",      type=str, required=True,
                   help="Directory with offline pre-collected scored JSONL files "
                        "(flat directory — all _scored.jsonl files will be loaded)")
    p.add_argument("--pilot_question_json", type=str, required=True,
                   help="JSON file with ALL pilot questions (MATH-500 format) "
                        "used as the resampling universe")
    p.add_argument("--min_group_size",      type=int, default=2)
    p.add_argument("--n_trajs_per_question",type=int, default=8)

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
    p.add_argument("--max_steps",    type=int,   default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--kl_coeff",     type=float, default=0.01)
    p.add_argument("--clip_ratio",   type=float, default=0.2)
    p.add_argument("--max_grad_norm",type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)

    # Checkpointing (frequent)
    p.add_argument("--checkpoint_every", type=int, default=50,
                   help="Save LoRA checkpoint every N training steps")

    # Resampling (less frequent, involves vLLM inference)
    p.add_argument("--resample_every",       type=int,   default=100,
                   help="Resample pool every N steps (also saves checkpoint)")
    p.add_argument("--n_resample_questions", type=int,   default=100,
                   help="Number of questions to resample per cycle (rotating window)")
    p.add_argument("--seeds_per_resample",   type=int,   default=2,
                   help="Number of seeds to use per resample cycle")
    p.add_argument("--resample_seeds",       type=int, nargs="+",
                   default=[42, 43, 44, 45, 46, 47, 48, 49],
                   help="Seed pool rotated across resampling cycles for diversity")
    p.add_argument("--max_agent_steps",      type=int,   default=5)
    p.add_argument("--vllm_tp_size",         type=int,   default=4,
                   help="vLLM tensor_parallel_size for resampling inference (default 4 = all GPUs)")

    # Data quality gate
    p.add_argument("--quality_min_acc", type=float, default=0.30,
                   help="Minimum accuracy (fraction correct) for initial pool quality check")

    # Checkpoint resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--initial_step",           type=int, default=0)

    # Output
    p.add_argument("--output_dir",   type=str,
                   default="training_outputs/qwen3-32B/os_rl_full_v3")
    p.add_argument("--log_every",    type=int, default=5)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
