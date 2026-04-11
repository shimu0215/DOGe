"""
resample_server.py — Persistent vLLM server for OS-RL training.

Architecture
------------
This process is launched from the SLURM job shell BEFORE the accelerate training
job.  It starts a persistent vLLM OpenAI-compatible server on GPU1/2/3 (tp=3),
waits for it to be healthy, then polls for resample requests.

On each resample cycle:
  1. Training rank-0 writes  {work_dir}/resample_request.json
  2. Server loads the new LoRA checkpoint via POST /v1/load_lora_adapter
  3. Runs per-seed inference subprocesses that connect to the HTTP server
     (no local GPU allocation — just HTTP requests)
  4. Unloads LoRA, writes {work_dir}/resample_done.json

Memory profile (ZeRO-3 CPU-offload + persistent vLLM on GPU1/2/3):
  GPU0 : training rank-0 only           ~3-8 GB (CPU-offload keeps GPU lean)
  GPU1 : vLLM (≈21 GB) + ZeRO-3 shard  ~24 GB peak  (81 GB total free)
  GPU2 : same as GPU1                   ~24 GB peak
  GPU3 : same as GPU1                   ~24 GB peak
  KV cache: (0.8×81 − 21) × 3 = ~135 GB across 3 GPUs (vs ~10 GB before)

File-signal protocol
---------------------
resample_request.json:
  {
    "checkpoint_dir"    : "/abs/path/to/lora-checkpoint",
    "step"              : <int>,
    "model_name"        : "Qwen/Qwen3-32B",
    "max_agent_steps"   : 5,
    "seeds_and_questions": [
      {"seed": 42, "questions": [{<qa-dict>}, ...]},
      ...
    ],
    "shutdown": false
  }

resample_done.json:
  {
    "step"         : <int>,
    "output_files" : ["/path/to/scored.jsonl", ...]
  }

server_ready: (empty file) written when vLLM server passes /health check.
  launch_dual.sh waits for this before starting training.
"""

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
import urllib.request
import urllib.error
import concurrent.futures as _cf
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

_REQUEST_FILE = "resample_request.json"
_DONE_FILE    = "resample_done.json"
_READY_FILE   = "server_ready"

# The LoRA adapter is always registered under this name on the vLLM server.
# VLLMServerModel in models.py hardcodes lora_name="finetune" → model="finetune"
# in every chat.completions request, so this must match.
_LORA_NAME    = "finetune"

_VLLM_PORT     = 8000
_VLLM_HOST     = "127.0.0.1"
_VLLM_BASE_URL = f"http://{_VLLM_HOST}:{_VLLM_PORT}"


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def _start_vllm_server(
    model_name: str,
    gpu_ids: str,
    gpu_memory_utilization: float,
    max_lora_rank: int,
    max_model_len: int,
    log_path: Path,
) -> subprocess.Popen:
    """
    Launch `vllm serve` as a background subprocess.

    CUDA_VISIBLE_DEVICES is set to gpu_ids (e.g. "1,2,3") so the server
    sees only those GPUs.  It starts with a clean CUDA context (no training
    ranks share this process), so tp > 1 works without the fork/ordinal error.
    """
    tp_size = len(gpu_ids.split(","))

    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES"        : gpu_ids,
        "HF_HUB_OFFLINE"              : "1",
        "TRANSFORMERS_OFFLINE"        : "1",
        "VLLM_HOST_IP"                : _VLLM_HOST,
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        "VLLM_NO_USAGE_STATS"         : "1",
        "DO_NOT_TRACK"                : "1",
    })
    for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
               "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RESTART_COUNT",
               "TORCHELASTIC_RUN_ID", "TORCHELASTIC_MAX_RESTARTS",
               "NCCL_ASYNC_ERROR_HANDLING", "VLLM_USE_V1"]:
        env.pop(_k, None)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_name,
        "--tensor-parallel-size",   str(tp_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--enable-lora",
        "--max-lora-rank",          str(max_lora_rank),
        "--port",                   str(_VLLM_PORT),
        "--host",                   _VLLM_HOST,
        "--max-model-len",          str(max_model_len),
        "--trust-remote-code",
        "--dtype",                  "bfloat16",
        "--disable-log-requests",
    ]
    logger.info(
        f"Launching vLLM server: tp={tp_size}, GPUs={gpu_ids}, "
        f"gpu_memory_utilization={gpu_memory_utilization}, "
        f"max_model_len={max_model_len}"
    )
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(_ROOT)
    )
    logger.info(f"vLLM server PID={proc.pid}, log={log_path}")
    return proc


def _wait_for_server(timeout: int = 900) -> bool:
    """Poll GET /health every 10 s until the server responds 200 or timeout."""
    url = f"{_VLLM_BASE_URL}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


# ---------------------------------------------------------------------------
# LoRA hot-swap helpers
# ---------------------------------------------------------------------------

def _api_post(path: str, payload: dict) -> int:
    """POST JSON to the vLLM server; return HTTP status code."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_VLLM_BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        logger.warning(f"API {path} → HTTP {e.code}: {body}")
        return e.code


def _load_lora(lora_path: str) -> bool:
    """Register the LoRA adapter at lora_path under name '{_LORA_NAME}'."""
    logger.info(f"Loading LoRA: {lora_path}")
    status = _api_post("/v1/load_lora_adapter", {
        "lora_name": _LORA_NAME,
        "lora_path": lora_path,
    })
    ok = status in (200, 201)
    if ok:
        logger.info(f"LoRA loaded (status={status}).")
    else:
        logger.warning(f"load_lora_adapter returned status={status}.")
    return ok


def _unload_lora() -> bool:
    """Unregister '{_LORA_NAME}' from the vLLM server."""
    status = _api_post("/v1/unload_lora_adapter", {"lora_name": _LORA_NAME})
    ok = status in (200, 201)
    if ok:
        logger.info("LoRA unloaded.")
    else:
        logger.warning(f"unload_lora_adapter returned status={status}.")
    return ok


# ---------------------------------------------------------------------------
# Per-seed inference via HTTP
# ---------------------------------------------------------------------------

def _run_one_seed(
    checkpoint_dir: str,
    step: int,
    seed: int,
    questions: list,
    model_name: str,
    max_agent_steps: int,
) -> list:
    """
    Run inference for one (seed, question-subset) pair.

    The subprocess does NOT load any model locally; it uses VLLMServerModel
    (OpenAI-compatible client) to talk to the persistent vLLM HTTP server.
    This means the subprocess itself uses virtually no GPU memory.

    Multiple seeds can run in parallel (ThreadPoolExecutor in serve()) because
    the vLLM server handles request batching via continuous batching.
    """
    log_root = Path(checkpoint_dir) / "qa_results"
    log_root.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"resample_step{step}_seed{seed}_",
        dir="/tmp",
        delete=False,
    )
    json.dump({"metadata": {"n": len(questions)}, "examples": questions}, tmp)
    tmp.close()

    # Subprocess env: no CUDA (just an HTTP client), no distributed vars
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE"    : "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    env.pop("CUDA_VISIBLE_DEVICES", None)   # subprocess uses no GPU directly
    for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
               "MASTER_ADDR", "MASTER_PORT",
               "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID",
               "TORCHELASTIC_MAX_RESTARTS", "NCCL_ASYNC_ERROR_HANDLING"]:
        env.pop(_k, None)

    try:
        cmd = [
            sys.executable, "-m",
            "exps_research.rl_training.run_with_file_dist",
            "--experiment_type",    "agent",
            "--task_type",          "math",
            "--data_path",          tmp.name,
            "--model_type",         "vllm",
            "--model_id",           model_name,
            "--fine_tuned",             # → VLLMServerModel with lora_name="finetune"
            "--lora_folder",        checkpoint_dir,  # used only for log_folder path
            "--log_folder",         str(log_root),
            "--n",                  "1",
            "--temperature",        "0.7",
            "--top_p",              "0.8",
            "--seed",               str(seed),
            "--max_steps",          str(max_agent_steps),
            "--search_engine_type", "python_only",
            "--suffix",             f"python_only_seed{seed}",
            "--parallel_workers",   "1",
            "--use_single_endpoint",    # → api_base = http://0.0.0.0:8000/v1
            # NO --use_local_model: that would load vLLM in-process (GPU-hungry)
        ]
        logger.info(
            f"  [seed={seed}] → HTTP server: {len(questions)} questions, step={step}"
        )
        result = subprocess.run(cmd, env=env, cwd=str(_ROOT))
        if result.returncode != 0:
            logger.warning(f"  [seed={seed}] subprocess rc={result.returncode}")
            return []
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

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

def serve(
    work_dir: str,
    model_name: str,
    gpu_ids: str = "1,2,3",
    gpu_memory_utilization: float = 0.8,
    max_lora_rank: int = 64,
    max_model_len: int = 24576,
    poll_interval: int = 5,
):
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    request_path = work / _REQUEST_FILE
    done_path    = work / _DONE_FILE
    ready_path   = work / _READY_FILE

    # ── Start persistent vLLM server ────────────────────────────────────────
    vllm_log  = work / "vllm_server.log"
    vllm_proc = _start_vllm_server(
        model_name=model_name,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=gpu_memory_utilization,
        max_lora_rank=max_lora_rank,
        max_model_len=max_model_len,
        log_path=vllm_log,
    )

    logger.info("Waiting for vLLM server (health check, timeout=900s)...")
    if not _wait_for_server(timeout=900):
        logger.error("vLLM server did not become healthy in 15 min. See vllm_server.log.")
        vllm_proc.kill()
        sys.exit(1)

    # Signal launch_dual.sh that training may now start
    ready_path.write_text("ready\n")
    logger.info(
        f"vLLM server ready. Wrote {ready_path}. "
        f"Watching {request_path} (poll every {poll_interval}s)"
    )

    try:
        while True:
            # ── Liveness check ──────────────────────────────────────────────
            if vllm_proc.poll() is not None:
                logger.error(f"vLLM server exited unexpectedly (rc={vllm_proc.returncode})!")
                sys.exit(1)

            if not request_path.exists():
                time.sleep(poll_interval)
                continue

            # ── Consume request ─────────────────────────────────────────────
            try:
                raw = request_path.read_text()
                request_path.unlink()
                request = json.loads(raw)
            except Exception as exc:
                logger.warning(f"Failed to read/delete request file: {exc}")
                time.sleep(poll_interval)
                continue

            if request.get("shutdown", False):
                logger.info("Shutdown signal received — exiting.")
                break

            step            = request.get("step", -1)
            checkpoint_dir  = request["checkpoint_dir"]
            model_name_req  = request.get("model_name", model_name)
            max_agent_steps = request.get("max_agent_steps", 5)
            seeds_and_qs    = request["seeds_and_questions"]

            _start = datetime.now()
            logger.info(
                f"[RESAMPLE START] step={step} "
                f"time={_start.strftime('%Y-%m-%d %H:%M:%S')} "
                f"seeds={[e['seed'] for e in seeds_and_qs]} "
                f"checkpoint={checkpoint_dir}"
            )

            # ── Hot-swap LoRA ────────────────────────────────────────────────
            _load_lora(checkpoint_dir)

            # ── Run seeds in parallel (all share the same HTTP server) ───────
            all_output_files = []
            with _cf.ThreadPoolExecutor(max_workers=len(seeds_and_qs)) as pool:
                futures = {
                    pool.submit(
                        _run_one_seed,
                        checkpoint_dir, step,
                        e["seed"], e["questions"],
                        model_name_req, max_agent_steps,
                    ): e["seed"]
                    for e in seeds_and_qs
                }
                for fut in _cf.as_completed(futures):
                    seed_val = futures[fut]
                    try:
                        files = fut.result()
                        all_output_files.extend(files)
                    except Exception as exc:
                        logger.warning(f"  [seed={seed_val}] raised: {exc}")

            # ── Unload LoRA so next cycle loads fresh weights ────────────────
            _unload_lora()

            # ── Write done signal ────────────────────────────────────────────
            _end     = datetime.now()
            elapsed  = (_end - _start).total_seconds()
            logger.info(
                f"[RESAMPLE DONE] step={step} "
                f"time={_end.strftime('%Y-%m-%d %H:%M:%S')} "
                f"elapsed={elapsed:.0f}s ({elapsed/60:.1f}min) "
                f"files={len(all_output_files)}"
            )
            tmp_done = done_path.with_suffix(".tmp")
            tmp_done.write_text(json.dumps({
                "step"        : step,
                "output_files": all_output_files,
                "elapsed_s"   : elapsed,
            }))
            tmp_done.rename(done_path)
            logger.info(f"Done signal written: {done_path}")

    finally:
        logger.info("Shutting down vLLM server...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Persistent vLLM server + LoRA hot-swap for OS-RL training"
    )
    p.add_argument(
        "--work_dir", required=True,
        help="Directory to watch for resample_request.json "
             "(should match --output_dir of the training script)"
    )
    p.add_argument(
        "--model_name", required=True,
        help="Base model name/path (e.g. Qwen/Qwen3-32B)"
    )
    p.add_argument(
        "--gpu_ids", default="1,2,3",
        help="Comma-separated physical GPU IDs for vLLM (default: 1,2,3)"
    )
    p.add_argument(
        "--gpu_memory_utilization", type=float, default=0.8,
        help="vLLM gpu_memory_utilization; training uses GPU0 and ZeRO-3 "
             "CPU-offload keeps GPU1/2/3 lean (~3-5 GB), so 0.8 leaves "
             "~57 GB free for KV cache per GPU (default: 0.8)"
    )
    p.add_argument(
        "--max_lora_rank", type=int, default=64,
        help="Maximum LoRA rank the server supports (must be >= training --lora_r, default: 64)"
    )
    p.add_argument(
        "--max_model_len", type=int, default=24576,
        help="Maximum sequence length for vLLM (default: 24576)"
    )
    p.add_argument(
        "--poll_interval", type=int, default=5,
        help="Seconds between polling attempts (default: 5)"
    )
    args = p.parse_args()

    serve(
        work_dir=args.work_dir,
        model_name=args.model_name,
        gpu_ids=args.gpu_ids,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        poll_interval=args.poll_interval,
    )
