import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
ERROR_MARKERS = (
    "AgentParsingError",
    "AgentExecutionError",
    "AgentMaxStepsError",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CodeAct outputs to SFT jsonl.")
    parser.add_argument("--input-dir", required=True, help="Directory containing gsm_hard_codeact_records*.jsonl")
    parser.add_argument("--output", required=True, help="Output jsonl path for SFT samples")
    parser.add_argument("--strict", action="store_true", help="Keep only success+correct samples and drop paths with any error step")
    parser.add_argument("--include-observation", action="store_true", help="Include tool observations in assistant training text")
    return parser.parse_args()


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(str(input_dir / "gsm_hard_codeact_records*.jsonl")))
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = THINK_TAG_RE.sub("", s)
    s = s.replace("\r\n", "\n")
    return s.strip()


def extract_observation_text(tool_result_text: str) -> str:
    """
    Prefer the concise final observed value if present, fallback to cleaned raw text.
    """
    s = clean_text(tool_result_text)
    m = re.search(r"Last output from code snippet:\s*(.*)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        last = m.group(1).strip()
        if last:
            return last
    return s


def has_error_noise(record: dict[str, Any]) -> bool:
    cot = str(record.get("cot_text", ""))
    if any(marker in cot for marker in ERROR_MARKERS):
        return True
    for step in record.get("trajectory", []):
        if step.get("step_type") == "error":
            return True
    return False


def build_segments(record: dict[str, Any]) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    step_idx = 0
    for step in record.get("trajectory", []):
        st = step.get("step_type")
        if st == "reasoning":
            text = clean_text(step.get("content", ""))
            if text:
                segments.append({"phase": "Thought", "text": text, "step_index": str(step_idx)})
                step_idx += 1
        elif st == "tool_call":
            code = clean_text(step.get("code", ""))
            if code:
                segments.append({"phase": "Code", "text": code, "step_index": str(step_idx)})
                step_idx += 1
        elif st == "tool_result":
            raw = clean_text(step.get("content", ""))
            if raw:
                segments.append({"phase": "Execution", "text": raw, "step_index": str(step_idx)})
                obs = extract_observation_text(raw)
                if obs:
                    segments.append({"phase": "Observation", "text": obs, "step_index": str(step_idx)})
                step_idx += 1
        elif st == "final_answer":
            final = clean_text(step.get("content", ""))
            if final:
                segments.append({"phase": "FinalAnswer", "text": final, "step_index": str(step_idx)})
                step_idx += 1
    return segments


def build_assistant_response(segments: list[dict[str, str]], include_observation: bool) -> str:
    parts: list[str] = []
    for seg in segments:
        phase = seg["phase"]
        text = seg["text"]
        if phase == "Thought":
            parts.append(f"Thought:\n{text}")
        elif phase == "Code":
            parts.append(f"Code:\n```python\n{text}\n```")
        elif phase == "Execution":
            # By default we don't train on tool execution traces.
            continue
        elif phase == "Observation":
            if include_observation:
                parts.append(f"Observation:\n{text}")
        elif phase == "FinalAnswer":
            parts.append(f"Final answer:\n{text}")
    return "\n\n".join(parts).strip()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_dir)
    total = len(records)
    kept = 0
    dropped_not_correct = 0
    dropped_not_success = 0
    dropped_error_noise = 0
    dropped_empty_response = 0

    with output_path.open("w", encoding="utf-8") as out:
        for r in records:
            if not r.get("correct", False):
                dropped_not_correct += 1
                continue
            if r.get("state") != "success":
                dropped_not_success += 1
                continue
            if args.strict and has_error_noise(r):
                dropped_error_noise += 1
                continue

            segments = build_segments(r)
            response = build_assistant_response(segments, include_observation=args.include_observation)
            if not response:
                dropped_empty_response += 1
                continue

            sample = {
                "id": f"gsmhard_q{r.get('question_id')}_s{r.get('sample_id')}",
                "question_id": r.get("question_id"),
                "sample_id": r.get("sample_id"),
                "messages": [
                    {"role": "user", "content": clean_text(r.get("question", ""))},
                    {"role": "assistant", "content": response},
                ],
                "segments": segments,
                "meta": {
                    "ground_truth": r.get("ground_truth"),
                    "extracted_ground_truth": r.get("extracted_ground_truth"),
                    "extracted_prediction": r.get("extracted_prediction"),
                    "state": r.get("state"),
                    "correct": r.get("correct"),
                    "token_usage": r.get("token_usage"),
                    "timing": r.get("timing"),
                },
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "input_dir": str(input_dir),
        "output": str(output_path),
        "strict": args.strict,
        "total_records": total,
        "kept_records": kept,
        "dropped_not_correct": dropped_not_correct,
        "dropped_not_success": dropped_not_success,
        "dropped_error_noise": dropped_error_noise,
        "dropped_empty_response": dropped_empty_response,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
