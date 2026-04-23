#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exps_research.analysis.plot_tool_call_prob_curve import (
    MarkerSpan,
    align_prefix,
    build_python_only_system_prompt,
    get_torch_dtype,
    load_rows,
    locate_marker_spans,
    make_truncated_messages,
    prepare_messages,
    sanitize_plot_text,
    shorten,
    tokenize_messages,
)
from exps_research.unified_framework.score_answers import evaluate_math_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot assistant-only P(next token = 'Code') curves and export high-probability contexts."
    )
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--target_text", type=str, default="Code")
    parser.add_argument("--context_radius", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def locate_assistant_token_spans(tokenizer, messages: Sequence[Dict[str, str]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    full_ids = tokenize_messages(tokenizer, messages)
    spans: List[Tuple[int, int]] = []
    for message_idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        before_messages = list(messages[:message_idx])
        through_messages = list(messages[: message_idx + 1])
        before_ids = tokenize_messages(tokenizer, before_messages) if before_messages else []
        through_ids = tokenize_messages(tokenizer, through_messages)
        token_start = align_prefix(full_ids, before_ids)
        token_end = align_prefix(full_ids, through_ids)
        if token_end > token_start:
            spans.append((token_start, token_end))
    return full_ids, spans


def assistant_prefix_positions(assistant_spans: Sequence[Tuple[int, int]]) -> List[int]:
    positions: List[int] = []
    for start, end in assistant_spans:
        positions.extend(range(start, end))
    return positions


def compute_next_token_curve(model, input_ids: List[int], target_token_id: int) -> List[float]:
    input_device = next(model.parameters()).device
    inputs = torch.tensor([input_ids], dtype=torch.long, device=input_device)
    with torch.no_grad():
        outputs = model(input_ids=inputs, use_cache=False)
    logits = outputs.logits[0, :-1].float()
    selected = logits[:, target_token_id]
    normalization = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(selected - normalization)
    return probs.cpu().tolist()


def count_action_steps(row: Dict) -> int:
    traces = row.get("log_data", {}).get("generation_trace", [])
    return sum(1 for trace in traces if trace.get("step_type") in {"action", "action_finalize"})


def compute_correctness(row: Dict) -> float:
    result = evaluate_math_answer(
        model=None,
        predicted=row.get("generated_answer"),
        gold=row.get("true_answer"),
        question=row.get("question"),
        do_extract_answer=False,
    )
    return float(result["score"])


def render_context(
    tokens: Sequence[str],
    peak_idx: int,
    left: int,
    right: int,
) -> List[str]:
    context = []
    for idx in range(left, right):
        token = tokens[idx].replace("\n", "\\n")
        if idx == peak_idx:
            context.append(f"<<<PEAK:{token}>>>")
        else:
            context.append(token)
    return context


def plot_one(
    output_path: Path,
    full_axis_positions: Sequence[int],
    full_axis_probs: Sequence[float | None],
    tool_call_starts: Sequence[int],
    row: Dict,
    step_count: int,
    correctness: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    plot_values = [float("nan") if value is None else value for value in full_axis_probs]
    ax.plot(full_axis_positions, plot_values, color="steelblue", linewidth=1.5, label="P(next token = 'Code')")
    for idx, pos in enumerate(tool_call_starts):
        ax.axvline(
            pos,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="actual tool-call start" if idx == 0 else None,
        )
    ax.set_xlabel("Full-token position across the cleaned full sequence")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"sample={row['_raw_index']} steps={step_count} "
        f"correct={'yes' if correctness > 0 else 'no'}"
    )
    question_preview = sanitize_plot_text(shorten(row.get("question", "")))
    answer_preview = sanitize_plot_text(shorten(row.get("generated_answer", ""), limit=90))
    text_box = (
        f"Question: {question_preview}\n"
        f"Answer: {answer_preview}\n"
        f"Tool-call starts: {list(tool_call_starts)}"
    )
    ax.text(
        0.01,
        0.98,
        text_box,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.trajectory_path, args.limit)
    if not rows:
        raise RuntimeError("No usable rows with log_data were found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_token_ids = tokenizer.encode(args.target_text, add_special_tokens=False)
    if len(target_token_ids) != 1:
        raise ValueError(
            f"Expected target_text={args.target_text!r} to map to one token, got ids={target_token_ids}"
        )
    target_token_id = target_token_ids[0]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=get_torch_dtype(args.torch_dtype),
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    system_prompt = build_python_only_system_prompt()
    summary_rows = []

    for local_idx, row in enumerate(rows):
        messages = prepare_messages(row, system_prompt)
        input_ids, assistant_spans = locate_assistant_token_spans(tokenizer, messages)
        _, tool_call_spans = locate_marker_spans(tokenizer, messages, "Code:")
        full_curve = compute_next_token_curve(model, input_ids, target_token_id)

        valid_positions = []
        valid_probs = []
        for pos in assistant_prefix_positions(assistant_spans):
            if pos - 1 < 0 or pos - 1 >= len(full_curve):
                continue
            valid_positions.append(pos)
            valid_probs.append(full_curve[pos - 1])

        assistant_prob_by_position = {pos: prob for pos, prob in zip(valid_positions, valid_probs)}
        full_axis_positions = list(range(1, len(input_ids)))
        full_axis_probs = [assistant_prob_by_position.get(pos) for pos in full_axis_positions]

        ranked = sorted(
            zip(valid_positions, valid_probs),
            key=lambda item: item[1],
            reverse=True,
        )[: args.top_k]

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        top_contexts = []
        for rank, (position, prob) in enumerate(ranked, start=1):
            left = max(0, position - args.context_radius)
            right = min(len(all_tokens), position + args.context_radius + 1)
            top_contexts.append(
                {
                    "rank": rank,
                    "position": position,
                    "prob": prob,
                    "peak_token": all_tokens[position].replace("\n", "\\n"),
                    "window_token_range": [left, right],
                    "context_tokens": render_context(all_tokens, position, left, right),
                }
            )

        step_count = count_action_steps(row)
        correctness = compute_correctness(row)
        figure_path = output_dir / f"trajectory_{local_idx:02d}_sample_{row['_raw_index']:04d}.png"
        plot_one(
            figure_path,
            full_axis_positions,
            full_axis_probs,
            [span.token_start for span in tool_call_spans],
            row,
            step_count,
            correctness,
        )

        summary_rows.append(
            {
                "local_index": local_idx,
                "raw_index": row["_raw_index"],
                "question": row.get("question", ""),
                "true_answer": row.get("true_answer", ""),
                "generated_answer": row.get("generated_answer", ""),
                "correct": correctness,
                "step_count": step_count,
                "assistant_token_spans": assistant_spans,
                "actual_tool_call_starts": [span.token_start for span in tool_call_spans],
                "target_text": args.target_text,
                "target_token_id": target_token_id,
                "curve_positions": valid_positions,
                "curve_probs": valid_probs,
                "top_contexts": top_contexts,
                "figure_path": str(figure_path),
            }
        )
        with (output_dir / "summary.json").open("w") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
        print(
            f"[{local_idx + 1}/{len(rows)}] sample={row['_raw_index']} "
            f"assistant_points={len(valid_positions)} top_prob={top_contexts[0]['prob'] if top_contexts else 0:.6f} "
            f"saved={figure_path}"
        )

    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
