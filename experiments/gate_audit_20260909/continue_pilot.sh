#!/usr/bin/env bash
set -euo pipefail
cd /scratch/wzhao20/opd-gate-audit-run-20260909
while kill -0 3781729 2>/dev/null; do sleep 10; done
export PYTHONPATH=/scratch/wzhao20/AKDA2/gsm_vocab_aligned_minillm_20260909:/scratch/wzhao20/opd-gate-audit-run-20260909/experiments/gate_audit_20260909
PY=/scratch/wzhao20/conda_envs/minillm_official/bin/python
"$PY" - <<'PY'
import json
from pathlib import Path
from context_audit import summarize
out=Path('results/context96')
rows=[json.loads(x) for x in (out/'scores.jsonl').read_text().splitlines()]
assert len(rows)==288, f'Incomplete diagnostic: {len(rows)}/288'
summarize(rows,out)
PY
"$PY" experiments/gate_audit_20260909/check_gate.py
srun --jobid=9762021 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  env AUDIT_ARM=legacy bash experiments/gate_audit_20260909/run_gate.sh > results/legacy_t120.log 2>&1
touch results/legacy_complete
