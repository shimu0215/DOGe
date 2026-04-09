"""
run_with_file_dist.py — Wrapper for run_experiment that patches
torch.distributed.init_process_group to use file:// (FileStore) instead of
tcp:// (TCPStore).

This fixes a SLURM issue where vLLM's UniProcExecutor calls
torch.distributed.init_process_group with a tcp:// init_method, but the
TCPStore server fails to start (or the client can't connect) when the resample
subprocess is launched from within an accelerate training job.

Root cause: UniProcExecutor._distributed_args() uses
  get_distributed_init_method(get_ip(), get_open_port()) → "tcp://127.0.0.1:{port}"
which triggers PyTorch's _tcp_rendezvous_handler → TCPStore(is_master=True).
The TCPStore server start fails in this environment, causing a 600s timeout.

Solution: Monkeypatch torch.distributed.init_process_group before vLLM is
initialised so that tcp:// init_method is replaced with file:// (FileStore),
which needs no network at all.  For tp=1 / pp=1 / dp=1 this is safe because
all NCCL collective calls are no-ops for world_size=1.
"""

import os
import sys
import tempfile

# ── Patch torch.distributed BEFORE any vllm import ────────────────────────
import torch.distributed as _td

_orig_init_pg = _td.init_process_group


def _file_init_process_group(*args, **kwargs):
    """Replace tcp:// init_method with file:// to avoid TCPStore in SLURM."""
    if not _td.is_initialized():
        init_method = kwargs.get("init_method", None)
        # Also handle positional; signature is (backend, init_method, ...)
        if init_method is None and len(args) >= 2:
            init_method = args[1]

        if isinstance(init_method, str) and init_method.startswith("tcp://"):
            tmp_file = tempfile.mktemp(
                prefix=f"/tmp/vllm_dist_{os.getpid()}_"
            )
            new_method = f"file://{tmp_file}"
            print(
                f"[run_with_file_dist] Replacing {init_method} → {new_method}",
                flush=True,
            )
            kwargs["init_method"] = new_method
            # If init_method was positional, move it to kwargs
            if len(args) >= 2 and isinstance(args[1], str):
                args = (args[0],) + args[2:]

    return _orig_init_pg(*args, **kwargs)


_td.init_process_group = _file_init_process_group
# ──────────────────────────────────────────────────────────────────────────

# Now run the actual experiment (imports vLLM lazily)
from exps_research.unified_framework.run_experiment import run_experiment  # noqa: E402

run_experiment()
