#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RAW_PATHS=()

usage() {
  cat <<'EOF'
Usage: check_quality.sh RAW_JSONL [RAW_JSONL ...]
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

setup_agentdistill_env

"$PYTHON_BIN" - "$@" <<'PY'
import json
import sys
from pathlib import Path

def inspect(path: Path):
    rows = 0
    unique_questions = set()
    empty_log = 0
    error_field = 0
    parse_error = 0
    execution_failed = 0
    reached_max_steps = 0
    null_generated = 0
    null_true = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                entry = json.loads(line)
            except Exception:
                continue

            question = entry.get("question") or entry.get("problem") or entry.get("prompt")
            if question:
                unique_questions.add(question)

            if entry.get("log_data") in (None, "", {}):
                empty_log += 1
            if "error" in entry:
                error_field += 1
            if entry.get("generated_answer") is None:
                null_generated += 1
            if entry.get("true_answer") is None:
                null_true += 1

            text = json.dumps(entry, ensure_ascii=False)
            if "Error in code parsing" in text:
                parse_error += 1
            if "Code execution failed" in text:
                execution_failed += 1
            if "Reached max steps" in text:
                reached_max_steps += 1

    print(f"FILE\t{path}")
    print(f"rows\t{rows}")
    print(f"unique_questions\t{len(unique_questions)}")
    print(f"empty_log\t{empty_log}")
    print(f"error_field\t{error_field}")
    print(f"parse_error\t{parse_error}")
    print(f"execution_failed\t{execution_failed}")
    print(f"reached_max_steps\t{reached_max_steps}")
    print(f"null_generated\t{null_generated}")
    print(f"null_true\t{null_true}")
    print()

for raw in sys.argv[1:]:
    inspect(Path(raw))
PY
