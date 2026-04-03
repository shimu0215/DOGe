"""
prepare_pilot_data.py — Sample N questions from full trajectory pool for pilot runs.

Usage:
    python prepare_pilot_data.py \
        --traj_dir  <path to scored JSONL files> \
        --output_dir <where to write filtered files> \
        --n_questions 50 \
        --seed_range 42 57 \
        --seed 0

Reads all *_scored.jsonl files for the given seeds, groups trajectories by
question, randomly samples N questions, then writes one filtered JSONL per
seed to output_dir.
"""

import os
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir",    type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--n_questions", type=int, default=50)
    p.add_argument("--seed_range",  type=int, nargs=2, default=[42, 57],
                   help="[start, end) seeds")
    p.add_argument("--seed",        type=int, default=0,
                   help="Random seed for question sampling")
    args = p.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_range[0], args.seed_range[1]))

    # 1. Collect all scored files, group all entries by question
    # We need to know which entries belong to which file (to preserve per-seed files)
    # Structure: per_seed[seed] = list of raw JSONL lines (str)
    per_seed: dict[int, list[str]] = {}
    question_to_seeds: dict[str, set] = defaultdict(set)

    for seed in seeds:
        matches = sorted(traj_dir.glob(f"*seed={seed}_*_scored.jsonl"))
        if not matches:
            logger.warning(f"No scored file found for seed={seed}")
            continue
        path = matches[0]
        lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = entry.get("question", "")
                if q:
                    question_to_seeds[q].add(seed)
                lines.append((q, line))
        per_seed[seed] = lines
        logger.info(f"seed={seed}: loaded {len(lines)} trajectories from {path.name}")

    # 2. Find questions that appear in at least 2 seeds (needed for GRPO grouping)
    all_questions = [q for q, s in question_to_seeds.items() if len(s) >= 2]
    logger.info(f"Total unique questions with ≥2 seeds: {len(all_questions)}")

    if len(all_questions) < args.n_questions:
        logger.warning(
            f"Only {len(all_questions)} questions available, using all of them."
        )
        args.n_questions = len(all_questions)

    # 3. Sample N questions
    random.seed(args.seed)
    sampled_qs = set(random.sample(all_questions, args.n_questions))
    logger.info(f"Sampled {len(sampled_qs)} questions.")

    # 4. Write filtered per-seed files
    total_written = 0
    for seed, lines in per_seed.items():
        # Find the original filename to mirror it in output
        matches = sorted(traj_dir.glob(f"*seed={seed}_*_scored.jsonl"))
        orig_name = matches[0].name if matches else f"seed={seed}_scored.jsonl"

        out_path = out_dir / orig_name
        written = 0
        with open(out_path, "w") as fout:
            for q, raw_line in lines:
                if q in sampled_qs:
                    fout.write(raw_line + "\n")
                    written += 1
        total_written += written
        logger.info(f"seed={seed}: wrote {written} trajectories → {out_path.name}")

    logger.info(f"Done. Total trajectories written: {total_written} "
                f"across {len(per_seed)} seed files.")

    # 5. Save question list for reproducibility
    q_list_path = out_dir / "sampled_questions.txt"
    with open(q_list_path, "w") as f:
        for q in sorted(sampled_qs):
            f.write(q[:120].replace("\n", " ") + "\n")
    logger.info(f"Question list saved to: {q_list_path}")


if __name__ == "__main__":
    main()
