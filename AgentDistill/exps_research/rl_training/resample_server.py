"""
resample_server.py — Persistent resample server for OS-RL training.

Architecture
------------
This process is launched from the SLURM job shell BEFORE (or alongside) the
accelerate training job.  Because it is a completely independent OS process
(not forked from any training rank), it has a CLEAN CUDA context and can
initialise vLLM with tp=4 across all 4 GPUs without the
"CUDA invalid device ordinal" issue that plagues rank-0-spawned subprocesses.

File-signal protocol
---------------------
Training rank-0 writes  {work_dir}/resample_request.json  to trigger a cycle.
Server writes           {work_dir}/resample_done.json      when finished.

resample_request.json
  {
    "checkpoint_dir"   : "/abs/path/to/lora-checkpoint",
    "step"             : <int>,
    "model_name"       : "Qwen/Qwen3-32B",
    "max_agent_steps"  : 5,
    "seeds_and_questions": [
      {"seed": 42, "questions": [ {<qa-dict>}, ... ]},
      ...
    ],
    "shutdown": false
  }

resample_done.json
  {
    "step"         : <int>,
    "output_files" : ["/path/to/scored.jsonl", ...]
  }

Memory notes
------------
- Training ZeRO-3 CPU-offload: while all ranks sit in barrier() the GPU holds
  almost no training tensors (~0–1 GB reserved cache).
- vLLM tp=4 on a clean context: 32B bf16 ≈ 16 GB/GPU + KV cache.  With
  gpu_memory_utilization=0.85 and max_model_len=24576 this fits comfortably in
  80 GB A100 GPUs.
- The two processes alternate; they never compute simultaneously.
"""

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# ── Strip any distributed env vars that might have leaked from the launcher ──
for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
           "MASTER_ADDR", "MASTER_PORT",
           "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
           "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_TIMEOUT_KEEP_ALIVE",
           "NCCL_ASYNC_ERROR_HANDLING"]:
    os.environ.pop(_k, None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [resample_server] %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]   # AgentDistill/

# File names used for coordination
_REQUEST_FILE = "resample_request.json"
_DONE_FILE    = "resample_done.json"
_POLL_INTERVAL_S = 5          # seconds between polling attempts


# ---------------------------------------------------------------------------
# Subprocess environment
# ---------------------------------------------------------------------------

def _build_vllm_env(gpu_id: int) -> dict:
    """
    Return a clean environment dict for one tp=1 vLLM inference subprocess.

    Each subprocess gets exactly ONE GPU (CUDA_VISIBLE_DEVICES=str(gpu_id)).
    This avoids the tp>1 multiprocessing issue:
      - vLLM's multiproc executor fork()s workers that inherit the parent's
        CUDA context (device 0).  When a forked worker tries set_device(1/2/3)
        it gets "invalid device ordinal".
      - With tp=1 and a single visible GPU, no worker forking occurs at all;
        vLLM runs entirely in-process.
    Multiple seeds run in PARALLEL (one subprocess per GPU), giving the same
    total throughput as tp=N but without any distributed CUDA setup.
    """
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE"              : "1",
        "TRANSFORMERS_OFFLINE"        : "1",
        "VLLM_HOST_IP"               : "127.0.0.1",
        # Pin this subprocess to exactly one GPU (no fork/spawn workers needed)
        "CUDA_VISIBLE_DEVICES"        : str(gpu_id),
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        # Ensure tp=1 / single-process vLLM (belt-and-suspenders)
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    })
    # Remove distributed-training vars so vLLM does not see them
    for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
               "MASTER_ADDR", "MASTER_PORT",
               "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
               "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_TIMEOUT_KEEP_ALIVE",
               "NCCL_ASYNC_ERROR_HANDLING",
               "VLLM_USE_V1"]:
        env.pop(_k, None)
    return env


# ---------------------------------------------------------------------------
# Single-seed resample via subprocess
# ---------------------------------------------------------------------------

def _run_one_seed(
    checkpoint_dir: str,
    step: int,
    seed: int,
    questions: list,
    model_name: str,
    max_agent_steps: int,
    gpu_id: int = 0,
) -> list:
    """
    Run run_with_file_dist for one (seed, question-subset) pair with tp=1.

    Each call is pinned to a single GPU via CUDA_VISIBLE_DEVICES=gpu_id.
    tp=1 avoids vLLM's multiprocessing worker fork/spawn which causes
    "invalid device ordinal" errors when CUDA contexts are already active.

    Multiple seeds can be run in parallel (one per GPU) by the caller.

    Returns list of scored JSONL file paths written by run_experiment.
    """
    log_root = Path(checkpoint_dir) / "qa_results"
    log_root.mkdir(parents=True, exist_ok=True)

    # Write the question subset to a temp JSON file that run_experiment can read
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"resample_step{step}_seed{seed}_",
        dir="/tmp",
        delete=False,
    )
    json.dump({"metadata": {"n": len(questions)}, "examples": questions}, tmp)
    tmp.close()

    env = _build_vllm_env(gpu_id=gpu_id)

    try:
        cmd = [
            sys.executable, "-m",
            "exps_research.rl_training.run_with_file_dist",
            "--experiment_type",      "agent",
            "--task_type",            "math",
            "--data_path",            tmp.name,
            "--model_type",           "vllm",
            "--model_id",             model_name,
            "--fine_tuned",
            "--lora_folder",          checkpoint_dir,
            "--use_local_model",
            "--log_folder",           str(log_root),
            "--n",                    "1",
            "--temperature",          "0.7",
            "--top_p",                "0.8",
            "--seed",                 str(seed),
            "--max_steps",            str(max_agent_steps),
            "--search_engine_type",   "python_only",
            "--suffix",               f"python_only_seed{seed}",
            "--parallel_workers",     "1",   # vLLM offline is not thread-safe
            "--max_model_len",        "24576",
            "--tensor_parallel_size", "1",   # always tp=1; parallelism via separate subprocesses
            # Limit vLLM GPU memory fraction.  Training holds ~3 GB on GPU1/2/3
            # (ZeRO-3 parameter shard) while in barrier.  Free ≈ 78 GB.
            # Qwen3-32B weights ≈ 62.5 GB + vLLM V1 buffers → target 0.9 × 81 ≈ 73 GB,
            # leaving 10 GB for KV cache with max_model_len=24576.
            "--gpu_memory_utilization", "0.9",
        ]
        logger.info(
            f"  [seed={seed}] Launching inference on GPU{gpu_id}: "
            f"{len(questions)} questions, tp=1, "
            f"checkpoint={checkpoint_dir}"
        )
        result = subprocess.run(cmd, env=env, cwd=str(_ROOT))
        if result.returncode != 0:
            logger.warning(
                f"  [seed={seed}] run_experiment returned rc={result.returncode}"
            )
            return []
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    # Locate the scored output file
    found = list(log_root.glob(
        f"**/*seed={seed}*python_only_seed{seed}*_scored.jsonl"
    ))
    found = [p for p in found if ".bak" not in p.name]
    paths = [str(p) for p in found]
    logger.info(f"  [seed={seed}] Found {len(paths)} scored file(s).")
    return paths


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def serve(work_dir: str, n_gpus: int = 4, poll_interval: int = _POLL_INTERVAL_S):
    """
    Poll `{work_dir}/resample_request.json` in a loop.

    When a request arrives:
      1. Read & delete the request file (atomic consume).
      2. Run seeds in PARALLEL, each pinned to its own GPU (tp=1 per seed).
         GPU assignment: seed_i → GPU (i % n_gpus).
         This gives the same effective throughput as tp=n_gpus but without
         any distributed CUDA setup or fork/spawn worker issues.
      3. Write `{work_dir}/resample_done.json` to unblock the training loop.

    Why parallel tp=1 instead of single tp=4?
      vLLM's multiproc executor fork()s workers that inherit the parent's
      CUDA context.  Forked workers trying set_device(1/2/3) get
      "CUDA error: invalid device ordinal".  tp=1 avoids this entirely.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    request_path = work / _REQUEST_FILE
    done_path    = work / _DONE_FILE

    logger.info(
        f"Resample server ready.  Watching: {request_path} "
        f"(poll every {poll_interval}s, n_gpus={n_gpus}, parallel tp=1 per seed)"
    )

    while True:
        if not request_path.exists():
            time.sleep(poll_interval)
            continue

        # ── Consume request ──────────────────────────────────────────────
        try:
            raw = request_path.read_text()
            request_path.unlink()           # delete before processing
            request = json.loads(raw)
        except Exception as exc:
            logger.warning(f"Failed to read/delete request file: {exc}")
            time.sleep(poll_interval)
            continue

        if request.get("shutdown", False):
            logger.info("Shutdown signal received — exiting.")
            break

        step              = request.get("step", -1)
        checkpoint_dir    = request["checkpoint_dir"]
        model_name        = request["model_name"]
        max_agent_steps   = request.get("max_agent_steps", 5)
        seeds_and_qs      = request["seeds_and_questions"]
        _start = datetime.now()
        logger.info(
            f"[RESAMPLE START] step={step} "
            f"time={_start.strftime('%Y-%m-%d %H:%M:%S')} "
            f"seeds={[e['seed'] for e in seeds_and_qs]} "
            f"checkpoint={checkpoint_dir}"
        )

        # ── Run seeds in PARALLEL (one subprocess per GPU) ───────────────
        # Assign GPU round-robin: seed_i → GPU (i % n_gpus).
        # Training is in barrier, so all n_gpus are ~80 GB free.
        import concurrent.futures as _cf
        all_output_files = []

        def _run_entry(idx_entry):
            idx, seed_entry = idx_entry
            # Skip GPU0: training rank-0 caches ~16 GB in PyTorch's CUDA allocator
            # after each training step (allgather residuals), leaving insufficient room
            # for the 32B model.  GPU1/2/3 each hold only ~3 GB, so we use those.
            gpu_id = (idx % max(n_gpus - 1, 1)) + 1
            return _run_one_seed(
                checkpoint_dir=checkpoint_dir,
                step=step,
                seed=seed_entry["seed"],
                questions=seed_entry["questions"],
                model_name=model_name,
                max_agent_steps=max_agent_steps,
                gpu_id=gpu_id,
            )

        with _cf.ThreadPoolExecutor(max_workers=len(seeds_and_qs)) as pool:
            futures = {
                pool.submit(_run_entry, (i, e)): e["seed"]
                for i, e in enumerate(seeds_and_qs)
            }
            for fut in _cf.as_completed(futures):
                seed_val = futures[fut]
                try:
                    files = fut.result()
                    all_output_files.extend(files)
                except Exception as exc:
                    logger.warning(f"  [seed={seed_val}] raised exception: {exc}")

        # ── Write done signal ────────────────────────────────────────────
        _end = datetime.now()
        elapsed = (_end - _start).total_seconds()
        logger.info(
            f"[RESAMPLE DONE] step={step} "
            f"time={_end.strftime('%Y-%m-%d %H:%M:%S')} "
            f"elapsed={elapsed:.0f}s ({elapsed/60:.1f}min) "
            f"files={len(all_output_files)}"
        )
        done_data = {
            "step"         : step,
            "output_files" : all_output_files,
            "elapsed_s"    : elapsed,
        }
        # Write atomically via a tmp file then rename
        tmp_done = done_path.with_suffix(".tmp")
        tmp_done.write_text(json.dumps(done_data))
        tmp_done.rename(done_path)
        logger.info(f"Done signal written: {done_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Persistent resample server for OS-RL training (dual-process-group)"
    )
    p.add_argument(
        "--work_dir", required=True,
        help="Directory to watch for resample_request.json signals "
             "(should match --output_dir of the training script)"
    )
    p.add_argument(
        "--n_gpus", type=int, default=4,
        help="Number of GPUs to spread seeds across (one tp=1 subprocess per GPU). "
             "Default 4 = use all 4 GPUs in parallel."
    )
    p.add_argument(
        "--poll_interval", type=int, default=_POLL_INTERVAL_S,
        help="Seconds between polling attempts (default 5)"
    )
    args = p.parse_args()

    serve(
        work_dir=args.work_dir,
        n_gpus=args.n_gpus,
        poll_interval=args.poll_interval,
    )
