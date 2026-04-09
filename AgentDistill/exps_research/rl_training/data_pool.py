"""
data_pool.py — Trajectory pool for offline GRPO training.

Loads pre-collected JSONL trajectory files, groups by question,
and provides batch sampling + pool refresh for periodic resampling.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrajectoryPool:
    """
    Manages a pool of agent trajectories grouped by question.

    Each entry in the pool is a dict from a scored JSONL file with fields:
        question:          str  — problem text (used as group key)
        generated_answer:  str  — model's final answer
        true_answer:       str  — gold answer
        score:             bool — True if correct
        log_data:          dict — messages, original_memory, etc.
    """

    def __init__(self, jsonl_files: List[str], min_group_size: int = 2):
        """
        Args:
            jsonl_files:    List of paths to scored JSONL trajectory files.
            min_group_size: Minimum number of trajectories per question
                            to include that question in training batches.
        """
        self.min_group_size = min_group_size
        self.pool: Dict[str, List[dict]] = defaultdict(list)
        self._load(jsonl_files)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, jsonl_files: List[str]) -> None:
        total = 0
        for path in jsonl_files:
            path = str(path)
            if not Path(path).exists():
                logger.warning(f"Trajectory file not found, skipping: {path}")
                continue
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not entry.get("log_data"):
                        continue
                    qkey = entry["question"]
                    self.pool[qkey].append(entry)
                    total += 1
        logger.info(
            f"Loaded {total} trajectories across {len(self.pool)} questions "
            f"(min_group_size={self.min_group_size})"
        )

    def refresh(self, jsonl_files: List[str]) -> None:
        """Replace the current pool with newly collected trajectories."""
        self.pool = defaultdict(list)
        self._load(jsonl_files)

    def add_files(self, jsonl_files: List[str]) -> None:
        """Merge new trajectory files INTO the existing pool (append, not replace).

        This is preferred over refresh() when resampling produces only 1 new
        trajectory per question (n=1), which is below min_group_size=2.
        By merging with the original offline pool, each question accumulates
        multiple trajectories over time and remains valid for GRPO training.
        """
        self._load(jsonl_files)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def valid_questions(self) -> List[str]:
        """Questions with at least min_group_size trajectories."""
        return [q for q, trajs in self.pool.items() if len(trajs) >= self.min_group_size]

    def __len__(self) -> int:
        return len(self.valid_questions)

    def stats(self) -> Dict:
        sizes = [len(v) for v in self.pool.values()]
        correct = sum(
            sum(1 for e in v if e.get("score", False))
            for v in self.pool.values()
        )
        total = sum(sizes)
        return {
            "n_questions": len(self.pool),
            "n_valid_questions": len(self.valid_questions),
            "total_trajectories": total,
            "correct_trajectories": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "avg_trajs_per_question": total / len(self.pool) if self.pool else 0.0,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_batch(
        self,
        n_questions: int,
        n_trajs_per_question: int,
        require_both_correct_and_wrong: bool = False,
    ) -> List[List[dict]]:
        """
        Sample a batch of question groups.

        Returns:
            List of groups. Each group is a list of trajectory dicts
            for the same question.
        """
        valid_qs = self.valid_questions
        if not valid_qs:
            raise ValueError("No valid questions in pool (too few trajectories per question).")

        n_questions = min(n_questions, len(valid_qs))
        selected_qs = random.sample(valid_qs, n_questions)

        batch = []
        for q in selected_qs:
            trajs = self.pool[q]

            if require_both_correct_and_wrong:
                correct = [t for t in trajs if t.get("score", False)]
                wrong = [t for t in trajs if not t.get("score", False)]
                if not correct or not wrong:
                    # Fall back to random sample
                    selected = random.sample(trajs, min(n_trajs_per_question, len(trajs)))
                else:
                    n_correct = max(1, n_trajs_per_question // 2)
                    n_wrong = n_trajs_per_question - n_correct
                    selected = (
                        random.sample(correct, min(n_correct, len(correct))) +
                        random.sample(wrong, min(n_wrong, len(wrong)))
                    )
            else:
                selected = random.sample(trajs, min(n_trajs_per_question, len(trajs)))

            batch.append(selected)

        return batch

    def iter_all_questions(
        self, n_trajs_per_question: int, shuffle: bool = True
    ):
        """
        Iterate over all valid questions, yielding groups.
        Used for one full "epoch" over the pool.
        """
        qs = list(self.valid_questions)
        if shuffle:
            random.shuffle(qs)
        for q in qs:
            trajs = self.pool[q]
            selected = random.sample(trajs, min(n_trajs_per_question, len(trajs)))
            yield selected
