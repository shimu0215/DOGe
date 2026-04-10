"""
train_iib_rl.py — Semi-online Input Invariance Breaking (IIB) RL.

Reward structure:
    R_total = R_task(τ_x) + λ_inv * R_inv(τ_x, aug_pool[x])
            - β * KL(π_θ ‖ π_base)          [handled by GRPOTrainer]

    R_task:  Binary correctness (1.0 if correct, 0.0 otherwise).
    R_inv:   Mean normalised code-edit-distance between the original trajectory
             τ_x and all available augmented trajectories τ_{x'} in the aug pool
             for the same question.  Range: [0, 1].  Higher = more divergent.

Anti-distillation intuition:
    The teacher is trained to produce measurably different code trajectories
    when its input question is slightly reworded (x vs x' = aug(x)).  A student
    observing both (x, τ_x) and (x', τ_{x'}) cannot learn a consistent policy
    because the teacher's reasoning path varies for semantically equivalent inputs.

Augmentation (augment_utils.py):
    Template-based instruction paraphrase: same numbers, different framing.
    No external LLM call required.  Cheap and deterministic.

GRPO updates:
    Only applied to the ORIGINAL-question group (orig_group).  Aug trajectories
    serve only as divergence targets for R_inv — they are not backpropped through.
    Reason: aug pool typically has only 1 trajectory per question (n=1 resample),
    which yields zero GRPO advantage after normalisation.  Training on orig_group
    already injects the divergence gradient through R_inv.

Semi-online loop:
    Same as train_os_rl_online_pilot.py:
    - Every resample_every steps: save LoRA checkpoint.
    - Rank 0 collects fresh orig trajectories AND fresh aug trajectories (using
      the next augmentation template to vary style across cycles).
    - replace_with_files() for orig pool; add_files()-style FIFO for aug pool.
    - All ranks wait at barrier before training continues.

GPU layout during resampling:
    Training  : 4 processes on GPUs 0-3 (ZeRO-3 CPU offload, ~1 GB each)
    vLLM      : tp=1, one GPU (~79 GB free) — called twice (orig then aug)
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
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from torch.optim import AdamW

from .data_pool import TrajectoryPool
from .rewards import compute_r_task, _normalised_edit_distance
from .message_utils import clean_messages_for_training, extract_code_blocks
from .grpo_trainer import GRPOTrainer
from .augment_utils import build_aug_examples, N_RECIPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]   # AgentDistill/


# ---------------------------------------------------------------------------
# Augmented trajectory pool
# ---------------------------------------------------------------------------

class AugTrajectoryPool:
    """
    Thin trajectory pool for augmented questions, keyed by ORIGINAL question text.

    Augmented trajectories are collected with the augmented question as the
    "question" field.  A mapping (aug_q → orig_q) is required to re-key them
    so they align with the original pool.

    min_group_size=1 because we typically have only 1 aug trajectory per question
    (n=1 resample) — they are used only to compute R_inv, not for GRPO grouping.
    """

    def __init__(self) -> None:
        self.pool: Dict[str, List[dict]] = defaultdict(list)

    def _add_jsonl(
        self,
        jsonl_files: List[str],
        aug_to_orig: Dict[str, str],
        fifo: bool,
    ) -> int:
        """Load entries from JSONL files, re-keying by orig question.

        Args:
            fifo: If True, remove oldest entry for the question before adding
                  (replace strategy).  If False, just append.
        Returns number of entries added.
        """
        n_added = 0
        for path in jsonl_files:
            path = str(path)
            if not Path(path).exists():
                logger.warning(f"Aug trajectory file not found, skipping: {path}")
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not entry.get("log_data"):
                        continue
                    aug_q = entry.get("question", "")
                    orig_q = aug_to_orig.get(aug_q, aug_q)
                    if fifo and self.pool[orig_q]:
                        self.pool[orig_q].pop(0)
                    self.pool[orig_q].append(entry)
                    n_added += 1
        return n_added

    def add_files(self, jsonl_files: List[str], aug_to_orig: Dict[str, str]) -> int:
        """Append new aug trajectories (used at init / first resample)."""
        return self._add_jsonl(jsonl_files, aug_to_orig, fifo=False)

    def replace_with_files(self, jsonl_files: List[str], aug_to_orig: Dict[str, str]) -> int:
        """FIFO replace: remove oldest aug trajectory before adding new one."""
        return self._add_jsonl(jsonl_files, aug_to_orig, fifo=True)

    def get(self, orig_question: str) -> List[dict]:
        """Return all aug trajectories for a given original question."""
        return self.pool.get(orig_question, [])

    def stats(self) -> dict:
        sizes = [len(v) for v in self.pool.values()]
        total = sum(sizes)
        correct = sum(
            sum(1 for e in v if e.get("score", False))
            for v in self.pool.values()
        )
        return {
            "n_questions": len(self.pool),
            "total_aug_trajectories": total,
            "correct_aug_trajectories": correct,
            "aug_accuracy": correct / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# IIB reward computation
# ---------------------------------------------------------------------------

def _code_string(entry: dict) -> str:
    """Extract concatenated code blocks from a trajectory entry."""
    raw = entry.get("log_data", {}).get("messages", [])
    cleaned = clean_messages_for_training(raw)
    if cleaned is None:
        return ""
    return "\n".join(extract_code_blocks(cleaned))


def compute_iib_rewards(
    orig_group: List[dict],
    aug_pool: AugTrajectoryPool,
    lambda_inv: float,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute IIB rewards for a group of original trajectories.

    For each τ_x in orig_group:
        R_inv(τ_x)  = mean_{τ_x' in aug_pool[x]} D(τ_x, τ_x')
        R_total(τ_x) = R_task(τ_x) + lambda_inv * R_inv(τ_x)

    D = normalised code-block Levenshtein edit distance (range [0, 1]).
    If aug_pool is empty for this question, R_inv = 0.

    Returns:
        r_total, r_task, r_inv  — each a list of floats of len(orig_group)
    """
    r_task = compute_r_task(orig_group)

    # Retrieve augmented trajectories for this question
    orig_q = orig_group[0].get("question", "")
    aug_trajs = aug_pool.get(orig_q)

    if not aug_trajs or lambda_inv == 0.0:
        r_inv = [0.0] * len(orig_group)
    else:
        aug_codes = [_code_string(t) for t in aug_trajs]
        r_inv = []
        for entry in orig_group:
            oc = _code_string(entry)
            dists = [_normalised_edit_distance(oc, ac) for ac in aug_codes]
            r_inv.append(float(np.mean(dists)) if dists else 0.0)

    r_total = [rt + lambda_inv * ri for rt, ri in zip(r_task, r_inv)]
    return r_total, r_task, r_inv


# ---------------------------------------------------------------------------
# Model / tokenizer builders (identical to online pilot)
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
    if getattr(args, "resume_from_checkpoint", None):
        from peft import PeftModel
        logger.info(f"Resuming LoRA adapter from: {args.resume_from_checkpoint}")
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True
        )
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
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
    """Save LoRA adapter checkpoint (ZeRO-3 compatible).  Identical to online pilot."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step}")
    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        import deepspeed, json as _json, re as _re
        lora_named_params = [(n, p) for n, p in unwrapped.named_parameters()
                             if "lora_" in n]
        lora_params = [p for _, p in lora_named_params]
        param_dict = {}
        with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=None):
            if accelerator.is_main_process:
                def _clean_key(name):
                    return _re.sub(
                        r'\.(lora_A|lora_B|lora_embedding_A|lora_embedding_B)'
                        r'\.([^.]+)\.',
                        r'.\1.',
                        name,
                    )
                param_dict = {
                    _clean_key(n): p.data.detach().cpu().clone()
                    for n, p in lora_named_params
                }
        if accelerator.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(param_dict, os.path.join(ckpt_dir, "adapter_model.bin"))
            peft_cfg = list(unwrapped.peft_config.values())[0]
            with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
                class _SetEncoder(_json.JSONEncoder):
                    def default(self, o):
                        return sorted(o) if isinstance(o, set) else super().default(o)
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


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

def _build_env() -> dict:
    """Strip distributed-training env vars so vLLM subprocess starts clean."""
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
              "MASTER_ADDR", "MASTER_PORT",
              "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
              "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_TIMEOUT_KEEP_ALIVE",
              "NCCL_ASYNC_ERROR_HANDLING", "VLLM_USE_V1"]:
        env.pop(k, None)
    return env


def _run_collection(
    args,
    data_path: str,
    log_root: Path,
    seed: int,
    suffix: str,
    env: dict,
) -> List[str]:
    """Run run_experiment for one seed and return scored JSONL paths."""
    cmd = [
        sys.executable, "-m", "exps_research.rl_training.run_with_file_dist",
        "--experiment_type",    "agent",
        "--task_type",          "math",
        "--data_path",          data_path,
        "--model_type",         "vllm",
        "--model_id",           args.model_name,
        "--fine_tuned",
        "--lora_folder",        args.current_ckpt_dir,   # set by caller
        "--use_local_model",
        "--log_folder",         str(log_root),
        "--n",                  "1",
        "--temperature",        "0.7",
        "--top_p",              "0.8",
        "--seed",               str(seed),
        "--max_steps",          str(args.max_agent_steps),
        "--search_engine_type", "python_only",
        "--suffix",             suffix,
        "--parallel_workers",   "1",
        "--max_model_len",      "24576",
    ]
    result = subprocess.run(cmd, env=env, cwd=str(_ROOT))
    if result.returncode != 0:
        logger.warning(f"run_experiment failed (seed={seed}, suffix={suffix}, "
                       f"rc={result.returncode})")
        return []

    found = list(log_root.glob(f"**/*seed={seed}*{suffix}*_scored.jsonl"))
    found = [p for p in found if ".bak" not in p.name]
    return [str(p) for p in found]


def resample_orig_and_aug(
    args,
    ckpt_dir: str,
    step: int,
    seed: int,
    questions_subset: List[dict],
    template_idx: int,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Collect fresh ORIGINAL and AUGMENTED trajectories for questions_subset.

    Args:
        questions_subset: List of question dicts (MATH-500 format) to resample.
        template_idx:     Which augmentation template to use this cycle.

    Returns:
        orig_files:    Scored JSONL paths for original questions.
        aug_files:     Scored JSONL paths for augmented questions.
        aug_to_orig:   Mapping aug_question → orig_question.
    """
    # Attach ckpt_dir on args so _run_collection can read it
    args.current_ckpt_dir = ckpt_dir

    log_root = Path(ckpt_dir) / "qa_results"
    log_root.mkdir(parents=True, exist_ok=True)
    env = _build_env()

    logger.info(f"Step {step}: resampling {len(questions_subset)} questions "
                f"(seed={seed}, template={template_idx})")

    # ---- Collect ORIGINAL trajectories ----
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"orig_step{step}_", dir="/tmp", delete=False
    ) as tf:
        json.dump({"metadata": {"n": len(questions_subset)},
                   "examples": questions_subset}, tf)
        orig_tmp = tf.name

    try:
        orig_files = _run_collection(
            args, orig_tmp, log_root, seed,
            suffix=f"iib_orig_seed{seed}", env=env,
        )
    finally:
        try:
            os.unlink(orig_tmp)
        except OSError:
            pass

    # ---- Build augmented questions ----
    aug_examples, aug_to_orig = build_aug_examples(
        questions_subset, template_idx, noise_ops=args.noise_ops
    )

    # Save mapping alongside results for reproducibility / debugging
    mapping_path = log_root / f"aug_to_orig_step{step}_seed{seed}.json"
    with open(mapping_path, "w") as f:
        json.dump(aug_to_orig, f)

    # ---- Collect AUGMENTED trajectories ----
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"aug_step{step}_", dir="/tmp", delete=False
    ) as tf:
        json.dump({"metadata": {"n": len(aug_examples)},
                   "examples": aug_examples}, tf)
        aug_tmp = tf.name

    try:
        aug_files = _run_collection(
            args, aug_tmp, log_root, seed,
            suffix=f"iib_aug_seed{seed}", env=env,
        )
    finally:
        try:
            os.unlink(aug_tmp)
        except OSError:
            pass

    logger.info(f"Step {step}: collected {len(orig_files)} orig files, "
                f"{len(aug_files)} aug files.")
    return orig_files, aug_files, aug_to_orig


# ---------------------------------------------------------------------------
# Data quality check
# ---------------------------------------------------------------------------

def check_data_quality(
    pool: TrajectoryPool,
    n_sample: int = 50,
    min_accuracy: float = 0.30,
) -> bool:
    """Sample n_sample questions from pool and verify accuracy >= min_accuracy."""
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
        f"[Quality check] {len(sampled)} questions, "
        f"{correct}/{total} correct, accuracy={acc:.1%} "
        f"(threshold={min_accuracy:.1%})"
    )
    if acc < min_accuracy:
        raise RuntimeError(
            f"Data quality check FAILED: accuracy {acc:.1%} < {min_accuracy:.1%}. "
            "Check trajectory files or lower --quality_min_acc."
        )
    logger.info("[Quality check] PASSED.")
    return True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Extend NCCL timeout: vLLM resampling (orig + aug) can take ~60 min per cycle
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
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

    # ---- Load initial orig pool ----
    traj_files = []
    d = Path(args.trajectory_dir)
    traj_files = [str(p) for p in sorted(d.glob("*_scored.jsonl"))]
    if not traj_files:
        traj_files = [str(p) for p in sorted(d.rglob("*_scored.jsonl"))]
    logger.info(f"Found {len(traj_files)} trajectory files.")

    pool = TrajectoryPool(traj_files, min_group_size=args.min_group_size)
    if is_main:
        logger.info(f"Initial pool: {pool.stats()}")

    # ---- Data quality gate ----
    if is_main:
        check_data_quality(pool, n_sample=50, min_accuracy=args.quality_min_acc)
    accelerator.wait_for_everyone()

    # ---- Aug pool (empty at start; populated after first resample) ----
    aug_pool = AugTrajectoryPool()

    # ---- Load pilot questions for resampling ----
    with open(args.pilot_question_json) as f:
        pilot_data = json.load(f)
    all_pilot_questions = pilot_data["examples"]
    logger.info(f"Pilot questions: {len(all_pilot_questions)} "
                f"(resample {args.n_resample_questions} per cycle).")

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
    global_step    = getattr(args, "initial_step", 0)
    optimizer_step = 0
    last_resample_step = global_step
    last_ckpt_step     = global_step
    resample_cycle     = 0

    logger.info(
        f"Starting IIB semi-online RL "
        f"(initial_step={global_step}, "
        f"resample_every={args.resample_every}, "
        f"checkpoint_every={args.checkpoint_every}, "
        f"n_resample_questions={args.n_resample_questions}, "
        f"lambda_inv={args.lambda_inv})"
    )

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        for orig_group in pool.iter_all_questions(
            n_trajs_per_question=args.n_trajs_per_question, shuffle=True
        ):
            if global_step >= args.max_steps:
                break

            # ---- IIB reward ----
            r_total, r_task, r_inv = compute_iib_rewards(
                orig_group, aug_pool, args.lambda_inv
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            info = grpo.step(
                batch_groups=[orig_group],
                rewards_per_group=[r_total],
                global_step=global_step,
            )

            global_step += 1
            if info["optimizer_step"]:
                optimizer_step += 1

            if is_main and global_step % args.log_every == 0:
                log_entry = {
                    "step":           global_step,
                    "optimizer_step": optimizer_step,
                    "loss":           info.get("loss", 0.0),
                    "r_task_mean":    float(sum(r_task) / max(len(r_task), 1)),
                    "r_inv_mean":     float(sum(r_inv)  / max(len(r_inv),  1)),
                    "r_total_mean":   float(sum(r_total) / max(len(r_total), 1)),
                    "aug_pool_size":  len(aug_pool.pool),
                }
                logger.info(json.dumps(log_entry))
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # ---- Checkpoint (frequent) ----
            _ckpt_due = (
                args.checkpoint_every > 0
                and global_step % args.checkpoint_every == 0
                and global_step > last_ckpt_step
            )
            if _ckpt_due:
                last_ckpt_step = global_step
                save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)

            # ---- Resample orig + aug (less frequent) ----
            _resample_due = (
                args.resample_every > 0
                and global_step % args.resample_every == 0
                and global_step > last_resample_step
            )
            if _resample_due:
                last_resample_step = global_step

                # Ensure checkpoint exists at this step
                if global_step != last_ckpt_step:
                    last_ckpt_step = global_step
                    ckpt_dir = save_checkpoint(
                        model, tokenizer, args.output_dir, global_step, accelerator
                    )
                else:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-step{global_step}")

                if is_main:
                    # Select question subset (rotating window for coverage)
                    n_q = args.n_resample_questions
                    start = (resample_cycle * n_q) % len(all_pilot_questions)
                    if start + n_q <= len(all_pilot_questions):
                        subset = all_pilot_questions[start: start + n_q]
                    else:
                        subset = (all_pilot_questions[start:]
                                  + all_pilot_questions[: (start + n_q) % len(all_pilot_questions)])

                    # Rotate seed and augmentation template each cycle
                    seed_idx      = resample_cycle % len(args.resample_seeds)
                    resample_seed = args.resample_seeds[seed_idx]
                    template_idx  = 1 + (resample_cycle % N_RECIPES)
                    resample_cycle += 1

                    logger.info(
                        f"Resample cycle {resample_cycle}: "
                        f"{len(subset)} questions, seed={resample_seed}, "
                        f"template={template_idx}"
                    )

                    orig_files, aug_files, aug_to_orig = resample_orig_and_aug(
                        args, ckpt_dir, global_step,
                        seed=resample_seed,
                        questions_subset=subset,
                        template_idx=template_idx,
                    )

                    # Update orig pool (FIFO replace)
                    if orig_files:
                        n_replaced = pool.replace_with_files(orig_files)
                        logger.info(f"Orig pool: replaced {n_replaced} trajectories. "
                                    f"Stats: {pool.stats()}")

                    # Update aug pool
                    if aug_files:
                        if resample_cycle == 1:
                            # First resample: append (pool was empty)
                            n_aug = aug_pool.add_files(aug_files, aug_to_orig)
                        else:
                            # Subsequent: FIFO replace to keep size stable
                            n_aug = aug_pool.replace_with_files(aug_files, aug_to_orig)
                        logger.info(f"Aug pool: {n_aug} trajectories updated. "
                                    f"Stats: {aug_pool.stats()}")

                accelerator.wait_for_everyone()

        if global_step >= args.max_steps:
            break

    # Final checkpoint
    if global_step > last_ckpt_step:
        save_checkpoint(model, tokenizer, args.output_dir, global_step, accelerator)
    if is_main:
        logger.info("IIB training complete.")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Semi-online IIB RL")

    # Data
    p.add_argument("--trajectory_dir",      type=str, required=True,
                   help="Directory with pre-collected scored JSONL files (flat)")
    p.add_argument("--pilot_question_json", type=str, required=True,
                   help="JSON with questions for resampling (MATH-500 format)")
    p.add_argument("--min_group_size",      type=int, default=2)
    p.add_argument("--n_trajs_per_question",type=int, default=8)

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--lora_r",     type=int, default=32)
    p.add_argument("--max_length", type=int, default=2048)

    # IIB augmentation
    p.add_argument("--noise_ops",  type=int, default=2,
                   help="Number of random noise ops per question (0 = noise off). "
                        "Ops are sampled from: trailing_zero, random_synonym, "
                        "filler_phrase, number_elaboration, punctuation, pct_of.")

    # IIB reward
    p.add_argument("--lambda_inv", type=float, default=1.0,
                   help="Weight for R_inv (trajectory divergence reward). "
                        "Typical range: 0.5–2.0.")

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

    # Checkpointing
    p.add_argument("--checkpoint_every", type=int, default=50)

    # Resampling
    p.add_argument("--resample_every",       type=int,   default=100,
                   help="Resample orig+aug every N steps")
    p.add_argument("--n_resample_questions", type=int,   default=50,
                   help="Questions to resample per cycle (~10%% of 500)")
    p.add_argument("--resample_seeds",       type=int, nargs="+",
                   default=[42, 43, 44, 45, 46],
                   help="Seeds rotated across cycles")
    p.add_argument("--max_agent_steps",      type=int, default=5)

    # Quality gate
    p.add_argument("--quality_min_acc", type=float, default=0.30)

    # Resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--initial_step",           type=int, default=0)

    # Output
    p.add_argument("--output_dir", type=str,
                   default="training_outputs/qwen3-32B/iib_rl")
    p.add_argument("--log_every",  type=int, default=5)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
