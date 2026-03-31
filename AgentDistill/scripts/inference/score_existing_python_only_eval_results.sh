#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
LOG_ROOT="${LOG_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_eval}"
SUMMARY_ROOT="${SUMMARY_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_eval/scoring_summaries}"
mkdir -p "$SUMMARY_ROOT"

DATASETS=(
  "gsm_hard_500_20250507_test"
  "math_500_20250414_test"
)
MODELS=(
  "Qwen2.5-0.5B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen3-0.6B"
  "Qwen3-1.7B"
  "Qwen3-4B"
  "Qwen3-8B"
)

python - <<'PY'
import json, os, subprocess
from pathlib import Path

log_root = Path(os.environ.get('LOG_ROOT', '/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_eval'))
summary_root = Path(os.environ.get('SUMMARY_ROOT', '/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_eval/scoring_summaries'))
summary_root.mkdir(parents=True, exist_ok=True)
datasets = ['gsm_hard_500_20250507_test', 'math_500_20250414_test']
models = ['Qwen2.5-0.5B-Instruct','Qwen2.5-1.5B-Instruct','Qwen3-0.6B','Qwen3-1.7B','Qwen3-4B','Qwen3-8B']
all_summary = {}
for dataset in datasets:
    ds_dir = log_root / dataset
    ds_summary = {}
    for model in models:
        raw = ds_dir / f'{model}_temp=0.0_seed=42_type=agent_steps=5_python_only.jsonl'
        if not raw.exists():
            ds_summary[model] = {'status': 'missing_raw'}
            continue
        cmd = [
            'python', '-m', 'exps_research.unified_framework.score_answers',
            '--log_files', str(raw),
            '--task_type', 'math',
            '--max_workers', '8'
        ]
        proc = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True)
        base = raw.stem
        eval_summary = ds_dir / 'evaluations' / f'evaluation_summary_{base}.json'
        entry = {
            'status': 'ok' if proc.returncode == 0 and eval_summary.exists() else 'failed',
            'raw_file': str(raw),
            'summary_file': str(eval_summary),
            'returncode': proc.returncode,
            'stdout_tail': proc.stdout[-1000:],
            'stderr_tail': proc.stderr[-1000:],
        }
        if eval_summary.exists():
            with open(eval_summary) as f:
                entry['metrics'] = json.load(f)
        ds_summary[model] = entry
    all_summary[dataset] = ds_summary
with open(summary_root / 'agent_python_only_eval_summary.json', 'w') as f:
    json.dump(all_summary, f, indent=2)
print(summary_root / 'agent_python_only_eval_summary.json')
PY
