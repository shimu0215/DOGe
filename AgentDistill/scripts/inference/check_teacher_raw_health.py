#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    rows = 0
    unique_questions = set()
    empty_log = 0
    error_field = 0
    parse_error = 0
    max_steps = 0
    exec_failed = 0
    null_generated = 0
    null_true = 0
    sample_empty = []
    sample_error = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            entry = json.loads(line)
            q = entry.get("question") or entry.get("problem") or entry.get("prompt")
            if q:
                unique_questions.add(q)
            if entry.get("log_data") is None:
                empty_log += 1
                if len(sample_empty) < 3:
                    sample_empty.append(
                        {
                            "question": q,
                            "error": entry.get("error"),
                            "generated_answer": entry.get("generated_answer"),
                            "true_answer": entry.get("true_answer"),
                        }
                    )
            if entry.get("error") is not None:
                error_field += 1
                if len(sample_error) < 3:
                    sample_error.append({"question": q, "error": entry.get("error")})
            text = json.dumps(entry, ensure_ascii=False)
            if "Error in code parsing" in text:
                parse_error += 1
            if "Reached max steps" in text:
                max_steps += 1
            if "Code execution failed" in text:
                exec_failed += 1
            if entry.get("generated_answer") is None:
                null_generated += 1
            if entry.get("true_answer") is None:
                null_true += 1

    print(
        json.dumps(
            {
                "rows": rows,
                "unique_questions": len(unique_questions),
                "empty_log": empty_log,
                "error_field": error_field,
                "parse_error": parse_error,
                "max_steps": max_steps,
                "exec_failed": exec_failed,
                "null_generated": null_generated,
                "null_true": null_true,
            },
            ensure_ascii=False,
        )
    )
    print("EMPTY_SAMPLES")
    for item in sample_empty:
        print(json.dumps(item, ensure_ascii=False))
    print("ERROR_SAMPLES")
    for item in sample_error:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
