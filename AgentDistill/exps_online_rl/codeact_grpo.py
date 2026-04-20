import argparse
import copy
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from exps_research.train_utils.preprocess import PROMPT_TEMPLATES, clean_messages
from exps_research.unified_framework.processors.agent import AgentExperimentProcessor
from exps_research.unified_framework.score_answers import evaluate_math_answer
from exps_research.unified_framework.models import setup_model
from smolagents import FinalAnswerTool
from smolagents.agents import populate_template


@dataclass
class TrajectoryRecord:
    trajectory_id: int
    question: str
    true_answer: str
    rollout_seed: int
    sample_idx: int
    group_idx: int
    generated_answer: str
    correct: float
    cleaned_messages: List[Dict[str, str]]
    action_traces: List[Dict]
    task_reward: float = 0.0
    step_reward: float = 0.0
    kl_reward: float = 0.0
    total_reward: float = 0.0
    advantage: float = 0.0
    error: Optional[str] = None


@dataclass
class StepSample:
    sample_id: int
    trajectory_id: int
    question: str
    context_messages: List[Dict[str, str]]
    action_text: str
    input_ids: List[int]
    attention_mask: List[int]
    action_mask: List[int]
    old_token_logprobs: Optional[List[float]]
    advantage: float = 0.0
    old_logprob_mean: float = 0.0
    rollout_kl: float = 0.0


class StepDataset(Dataset):
    def __init__(self, step_samples: Sequence[StepSample]):
        self.step_samples = list(step_samples)

    def __len__(self) -> int:
        return len(self.step_samples)

    def __getitem__(self, idx: int) -> StepSample:
        return self.step_samples[idx]


def collate_step_samples(samples: Sequence[StepSample], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(sample.input_ids) for sample in samples)
    input_ids = []
    attention_mask = []
    action_mask = []
    old_logprob_mean = []
    advantages = []
    sample_ids = []
    trajectory_ids = []
    for sample in samples:
        pad = max_len - len(sample.input_ids)
        input_ids.append(sample.input_ids + [pad_token_id] * pad)
        attention_mask.append(sample.attention_mask + [0] * pad)
        action_mask.append(sample.action_mask + [0] * pad)
        old_logprob_mean.append(sample.old_logprob_mean)
        advantages.append(sample.advantage)
        sample_ids.append(sample.sample_id)
        trajectory_ids.append(sample.trajectory_id)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "action_mask": torch.tensor(action_mask, dtype=torch.bool),
        "old_logprob_mean": torch.tensor(old_logprob_mean, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "sample_ids": torch.tensor(sample_ids, dtype=torch.long),
        "trajectory_ids": torch.tensor(trajectory_ids, dtype=torch.long),
    }


def sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return [sanitize_for_json(v) for v in sorted(obj, key=lambda item: str(item))]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return sanitize_for_json(obj.item())
        except Exception:
            pass
    return str(obj)


class RolloutTimeoutError(RuntimeError):
    pass


def _rollout_timeout_handler(signum, frame):
    raise RolloutTimeoutError("rollout timed out")


class RolloutServerManager:
    def __init__(self, args):
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self.current_adapter_path: Optional[str] = None

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.args.rollout_port}/v1"

    def _build_cmd(self, adapter_path: Optional[str]) -> List[str]:
        cmd = [
            self.args.python_bin,
            "serve_vllm.py",
            "--model",
            self.args.model_name,
            "--port",
            str(self.args.rollout_port),
            "--tensor-parallel-size",
            str(self.args.rollout_tp),
            "--gpu-memory-utilization",
            str(self.args.rollout_gpu_util),
            "--max-model-len",
            str(self.args.rollout_max_length),
            "--disable-log-requests",
            "--disable-log-stats",
        ]
        if adapter_path:
            cmd.extend(["--lora-modules", f"finetune={adapter_path}", "--max-lora-rank", str(self.args.lora_r)])
        return cmd

    def _wait_until_ready(self, log_path: Path, timeout_s: int = 1800) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if log_path.exists():
                content = log_path.read_text(errors="ignore")
                if "Application startup complete." in content:
                    return
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {self.process.returncode}")
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for rollout server startup. See {log_path}")

    def start(self, adapter_path: Optional[str], run_dir: Path) -> None:
        self.stop()
        serve_log = run_dir / "rollout_server.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.args.rollout_cuda_visible_devices
        env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{env.get('PYTHONPATH', '')}"
        cmd = self._build_cmd(adapter_path)
        with serve_log.open("a") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {' '.join(cmd)}\n")
        log_fh = serve_log.open("a")
        self.process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        self.current_adapter_path = adapter_path
        self._wait_until_ready(serve_log)

    def restart(self, adapter_path: Optional[str], run_dir: Path) -> None:
        if adapter_path == self.current_adapter_path and self.process and self.process.poll() is None:
            return
        self.start(adapter_path, run_dir)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None


def load_math_entries(data_path: str) -> List[Dict]:
    with open(data_path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "examples" in payload:
        return list(payload["examples"])
    return list(payload)


def build_codeact_system_prompt(search_engine_type: str = "python_only") -> str:
    prompt_template = PROMPT_TEMPLATES["system_prompt_short"]
    tools = {}
    if search_engine_type != "python_only":
        raise NotImplementedError("This online GRPO entrypoint currently supports python_only rollouts only.")
    tools.setdefault("final_answer", FinalAnswerTool())
    return populate_template(prompt_template, variables={"tools": tools})


def normalize_messages_for_training(log_data: Dict, system_prompt: str) -> List[Dict[str, str]]:
    messages = clean_messages(copy.deepcopy(log_data["messages"]))
    if not messages:
        return messages
    messages[0]["content"] = system_prompt
    return messages


def extract_action_traces(log_data: Dict) -> List[Dict]:
    traces = []
    for trace in log_data.get("generation_trace", []):
        if trace.get("step_type") in {"action", "action_finalize"}:
            traces.append(trace)
    return traces


def align_prefix(full_ids: List[int], prompt_ids: List[int]) -> int:
    if len(prompt_ids) <= len(full_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    trimmed = list(prompt_ids)
    while trimmed and len(trimmed) <= len(full_ids):
        trimmed = trimmed[:-1]
        if full_ids[: len(trimmed)] == trimmed:
            return len(trimmed)
    raise ValueError("Unable to align prompt tokens with full tokens.")


def tokenize_step(
    tokenizer,
    context_messages: List[Dict[str, str]],
    action_text: str,
    max_length: int,
) -> Tuple[List[int], List[int], List[int]]:
    prompt_ids = tokenizer.apply_chat_template(
        context_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_messages = context_messages + [{"role": "assistant", "content": action_text}]
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prefix_len = align_prefix(full_ids, prompt_ids)
    if prefix_len >= len(full_ids):
        raise ValueError("Assistant action produced empty token span.")
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        prefix_len = max(prefix_len - overflow, 0)
        if prefix_len == 0 or prefix_len >= len(full_ids):
            raise ValueError("Prompt overflow removed the assistant span.")
    attention_mask = [1] * len(full_ids)
    action_mask = [0] * len(full_ids)
    for idx in range(prefix_len, len(full_ids)):
        action_mask[idx] = 1
    return full_ids, attention_mask, action_mask


def gather_step_samples(
    trajectories: Sequence[TrajectoryRecord],
    tokenizer,
    max_length: int,
) -> List[StepSample]:
    samples: List[StepSample] = []
    sample_id = 0
    for traj in trajectories:
        assistant_positions = [
            i
            for i, msg in enumerate(traj.cleaned_messages)
            if msg.get("role") == "assistant" and "Code:" in str(msg.get("content", ""))
        ]
        if not assistant_positions:
            continue
        for local_idx, msg_idx in enumerate(assistant_positions):
            context_messages = copy.deepcopy(traj.cleaned_messages[:msg_idx])
            action_text = traj.cleaned_messages[msg_idx]["content"]
            try:
                input_ids, attention_mask, action_mask = tokenize_step(
                    tokenizer=tokenizer,
                    context_messages=context_messages,
                    action_text=action_text,
                    max_length=max_length,
                )
            except ValueError:
                continue
            trace = traj.action_traces[local_idx] if local_idx < len(traj.action_traces) else {}
            rollout_logprobs = trace.get("generation_logprobs") or []
            old_token_logprobs = [float(item["logprob"]) for item in rollout_logprobs if item.get("logprob") is not None]
            samples.append(
                StepSample(
                    sample_id=sample_id,
                    trajectory_id=traj.trajectory_id,
                    question=traj.question,
                    context_messages=context_messages,
                    action_text=action_text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    old_token_logprobs=old_token_logprobs if old_token_logprobs else None,
                )
            )
            sample_id += 1
    return samples


def compute_masked_logprob_mean(logits: torch.Tensor, input_ids: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = action_mask[:, 1:]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    selected = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    lengths = shift_mask.sum(dim=-1).clamp(min=1)
    masked_sum = (selected * shift_mask).sum(dim=-1)
    return masked_sum / lengths


def compute_masked_forward_kl(
    current_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    current_shift = current_logits[:, :-1, :].float()
    ref_shift = ref_logits[:, :-1, :].float()
    shift_mask = action_mask[:, 1:]
    current_log_probs = F.log_softmax(current_shift, dim=-1)
    current_probs = current_log_probs.exp()
    ref_log_probs = F.log_softmax(ref_shift, dim=-1)
    token_kl = (current_probs * (current_log_probs - ref_log_probs)).sum(dim=-1)
    lengths = shift_mask.sum(dim=-1).clamp(min=1)
    masked_kl = (token_kl * shift_mask).sum(dim=-1) / lengths
    return masked_kl


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rollout_one_trajectory(
    entry: Dict,
    sample_idx: int,
    rollout_seed: int,
    args,
    api_base: str,
) -> Dict:
    enriched_entry = {"question": entry["question"], "answer": entry.get("answer")}
    retry_max_tokens = []
    if args.max_tokens is None:
        retry_max_tokens = [None]
    else:
        current_max_tokens = int(args.max_tokens)
        while current_max_tokens >= 32:
            retry_max_tokens.append(current_max_tokens)
            current_max_tokens //= 2

    last_result = None
    for max_tokens in retry_max_tokens:
        model_kwargs = {
            "model_type": "vllm",
            "model_id": args.model_name,
            "api_base": api_base,
            "api_key": "token-abc",
            "temperature": args.temperature,
            "seed": rollout_seed,
            "n": 1,
            "top_p": args.top_p,
        }
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        processor = AgentExperimentProcessor(model_kwargs, verbose=False)
        model = setup_model(**model_kwargs)
        timeout_s = max(int(args.rollout_timeout_seconds), 0)
        previous_handler = None
        if timeout_s > 0:
            previous_handler = signal.signal(signal.SIGALRM, _rollout_timeout_handler)
            signal.alarm(timeout_s)
        try:
            result = processor.process_entry(
                enriched_entry,
                model,
                search_engine_type="python_only",
                max_steps=args.max_steps,
                fine_tuned=False,
                verbose_worker=False,
            )
        except RolloutTimeoutError:
            result = {
                "question": entry["question"],
                "true_answer": entry.get("answer", ""),
                "generated_answer": "",
                "log_data": None,
                "error": f"rollout timeout after {timeout_s}s",
            }
        finally:
            if timeout_s > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
        result["sample_idx"] = sample_idx
        result["rollout_seed"] = rollout_seed
        result["used_max_tokens"] = max_tokens
        last_result = result
        error_text = str(result.get("error", ""))
        if "max_tokens" in error_text or "max_completion_tokens" in error_text:
            continue
        return result
    return last_result


def collect_rollouts_for_batch(
    entries: Sequence[Dict],
    group_offset: int,
    args,
    api_base: str,
    run_dir: Path,
    global_sync_idx: int,
) -> List[TrajectoryRecord]:
    raw_path = run_dir / f"rollouts_sync_{global_sync_idx:04d}.jsonl"
    futures = {}
    trajectory_id = 0
    trajectories: List[TrajectoryRecord] = []
    system_prompt = build_codeact_system_prompt("python_only")
    total_expected = len(entries) * args.num_rollouts_per_question

    def build_error_result(meta: Dict, error_text: str) -> Dict:
        return {
            "question": meta["entry"]["question"],
            "true_answer": meta["entry"].get("answer", ""),
            "generated_answer": "",
            "log_data": None,
            "error": error_text,
            "sample_idx": meta["sample_idx"],
            "rollout_seed": meta["rollout_seed"],
        }

    def append_trajectory(result: Dict) -> None:
        nonlocal trajectory_id
        question = result.get("question", "")
        group_idx = next(
            idx for idx, item in enumerate(entries) if item["question"] == question
        )
        correct = 0.0
        if not result.get("error"):
            eval_result = evaluate_math_answer(
                model=None,
                predicted=result.get("generated_answer"),
                gold=result.get("true_answer"),
                question=question,
                do_extract_answer=False,
            )
            correct = float(eval_result["score"])
        log_data = result.get("log_data")
        cleaned_messages = normalize_messages_for_training(log_data, system_prompt) if log_data else []
        action_traces = extract_action_traces(log_data) if log_data else []
        trajectories.append(
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                question=question,
                true_answer=str(result.get("true_answer")),
                rollout_seed=int(result["rollout_seed"]),
                sample_idx=int(result["sample_idx"]),
                group_idx=group_offset + group_idx,
                generated_answer=result.get("generated_answer", ""),
                correct=correct,
                cleaned_messages=cleaned_messages,
                action_traces=action_traces,
                task_reward=correct,
                total_reward=correct,
                error=result.get("error"),
            )
        )
        trajectory_id += 1

    def terminate_executor(executor: ProcessPoolExecutor) -> None:
        for proc in getattr(executor, "_processes", {}).values():
            if proc.is_alive():
                proc.terminate()
        executor.shutdown(wait=False, cancel_futures=True)

    executor = ProcessPoolExecutor(max_workers=args.rollout_workers)
    timed_out_batch = False
    try:
        for group_idx, entry in enumerate(entries):
            for sample_idx in range(args.num_rollouts_per_question):
                rollout_seed = args.seed + global_sync_idx * 100000 + group_idx * 100 + sample_idx
                future = executor.submit(
                    rollout_one_trajectory,
                    entry,
                    sample_idx,
                    rollout_seed,
                    args,
                    api_base,
                )
                futures[future] = {
                    "entry": entry,
                    "sample_idx": sample_idx,
                    "rollout_seed": rollout_seed,
                    "submitted_at": time.time(),
                }
        pending = set(futures)
        with raw_path.open("w") as f:
            while pending:
                done, not_done = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                expired = []
                for future in list(not_done):
                    meta = futures[future]
                    if now - meta["submitted_at"] > args.rollout_timeout_seconds:
                        expired.append(future)
                for future in done:
                    pending.remove(future)
                    meta = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = build_error_result(meta, f"rollout worker exception: {exc}")
                    f.write(json.dumps(sanitize_for_json(result), ensure_ascii=False) + "\n")
                    append_trajectory(result)
                if expired:
                    timed_out_batch = True
                    timeout_error = (
                        f"rollout batch aborted after worker timeout > {args.rollout_timeout_seconds}s"
                    )
                    for future in list(pending):
                        meta = futures[future]
                        result = build_error_result(meta, timeout_error)
                        f.write(json.dumps(sanitize_for_json(result), ensure_ascii=False) + "\n")
                        append_trajectory(result)
                    pending.clear()
                    f.flush()
                    terminate_executor(executor)
                    break
    finally:
        if not timed_out_batch:
            executor.shutdown(wait=True, cancel_futures=True)

    no_error_count = sum(1 for traj in trajectories if not traj.error)
    correct_count = sum(1 for traj in trajectories if traj.correct > 0)
    print(
        f"[sync {global_sync_idx}] rollout collected {len(trajectories)}/{total_expected} "
        f"trajectories, no_error={no_error_count}, correct={correct_count}"
    )
    trajectories.sort(key=lambda item: (item.group_idx, item.sample_idx, item.rollout_seed))
    return trajectories


def compute_step_statistics(
    accelerator: Accelerator,
    model,
    ref_model,
    step_samples: Sequence[StepSample],
    tokenizer,
    args,
) -> None:
    loader = DataLoader(
        StepDataset(step_samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_step_samples(batch, tokenizer.pad_token_id),
    )
    loader = accelerator.prepare(loader)
    gathered_old: Dict[int, float] = {}
    gathered_kl: Dict[int, float] = {}
    compute_kl = args.reward_mode == "task_kl"
    model.eval()
    if compute_kl and ref_model is not None:
        ref_model.eval()
    for batch in loader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            old_logprob_mean = compute_masked_logprob_mean(
                logits=outputs.logits,
                input_ids=batch["input_ids"],
                action_mask=batch["action_mask"],
            )
            sample_ids = batch["sample_ids"]
            if compute_kl:
                ref_outputs = ref_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                rollout_kl = compute_masked_forward_kl(
                    current_logits=outputs.logits,
                    ref_logits=ref_outputs.logits,
                    action_mask=batch["action_mask"],
                )
            else:
                rollout_kl = torch.zeros_like(old_logprob_mean)
            packed = torch.stack([sample_ids.to(torch.float32), old_logprob_mean, rollout_kl], dim=-1)
            gathered = accelerator.gather_for_metrics(packed)
            if accelerator.is_main_process:
                for row in gathered.cpu().tolist():
                    gathered_old[int(row[0])] = float(row[1])
                    gathered_kl[int(row[0])] = float(row[2])
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        for sample in step_samples:
            sample.old_logprob_mean = gathered_old[sample.sample_id]
            sample.rollout_kl = gathered_kl[sample.sample_id]
            if sample.old_token_logprobs and len(sample.old_token_logprobs) > 0:
                # Prefer the exact rollout-side value when token counts line up reasonably.
                sample.old_logprob_mean = sum(sample.old_token_logprobs) / len(sample.old_token_logprobs)


def assign_rewards_and_advantages(
    trajectories: Sequence[TrajectoryRecord],
    step_samples: Sequence[StepSample],
    args,
) -> None:
    step_count_by_trajectory: Dict[int, int] = {}
    kl_by_trajectory: Dict[int, List[float]] = {}
    for sample in step_samples:
        step_count_by_trajectory[sample.trajectory_id] = step_count_by_trajectory.get(sample.trajectory_id, 0) + 1
        if args.reward_mode == "task_kl":
            kl_by_trajectory.setdefault(sample.trajectory_id, []).append(sample.rollout_kl)
    for traj in trajectories:
        if args.reward_mode == "task_multistep":
            step_count = min(step_count_by_trajectory.get(traj.trajectory_id, 0), args.max_steps)
            if step_count > 1:
                traj.step_reward = 1.0 + 0.3 * max(step_count - 2, 0)
            else:
                traj.step_reward = 0.0
            traj.kl_reward = 0.0
            traj.total_reward = traj.task_reward + traj.step_reward
        else:
            step_kls = kl_by_trajectory.get(traj.trajectory_id, [])
            if args.kl_aggregation == "sum":
                traj.kl_reward = sum(step_kls)
            else:
                traj.kl_reward = sum(step_kls) / max(len(step_kls), 1)
            traj.step_reward = 0.0
            if traj.task_reward > 0:
                traj.total_reward = traj.task_reward + args.kl_lambda * traj.kl_reward
            else:
                traj.kl_reward = 0.0
                traj.total_reward = 0.0
    grouped: Dict[int, List[TrajectoryRecord]] = {}
    for traj in trajectories:
        grouped.setdefault(traj.group_idx, []).append(traj)
    for group in grouped.values():
        rewards = [traj.total_reward for traj in group]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / max(len(rewards), 1)
        std_reward = math.sqrt(max(variance, 1e-8))
        for traj in group:
            traj.advantage = (traj.total_reward - mean_reward) / std_reward
    traj_advantage = {traj.trajectory_id: traj.advantage for traj in trajectories}
    for sample in step_samples:
        sample.advantage = traj_advantage[sample.trajectory_id]


def save_lora_checkpoint(accelerator: Accelerator, model, tokenizer, output_dir: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()


def build_trainable_model(args):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    if args.resume_from_adapter:
        model = PeftModel.from_pretrained(base_model, args.resume_from_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    return model


def build_reference_model(args):
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    ref_model.config.use_cache = False
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def train_one_sync(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    step_samples: Sequence[StepSample],
    tokenizer,
    args,
    sync_idx: int,
    run_dir: Path,
) -> Dict[str, float]:
    train_loader = DataLoader(
        StepDataset(step_samples),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_step_samples(batch, tokenizer.pad_token_id),
    )
    train_loader = accelerator.prepare(train_loader)
    model.train()
    total_loss = 0.0
    total_ratio = 0.0
    total_batches = 0
    for epoch_idx in range(args.grpo_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                current_logprob_mean = compute_masked_logprob_mean(
                    logits=outputs.logits,
                    input_ids=batch["input_ids"],
                    action_mask=batch["action_mask"],
                )
                log_ratio = current_logprob_mean - batch["old_logprob_mean"]
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                advantages = batch["advantages"]
                loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu().item())
            total_ratio += float(ratio.detach().mean().cpu().item())
            total_batches += 1
        if accelerator.is_main_process:
            print(
                f"[sync {sync_idx}] finished epoch {epoch_idx + 1}/{args.grpo_epochs} "
                f"with mean loss={total_loss / max(total_batches, 1):.4f}"
            )
    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "ratio": total_ratio / max(total_batches, 1),
        "batches": float(total_batches),
    }
    if accelerator.is_main_process:
        metrics_path = run_dir / "train_metrics.jsonl"
        with metrics_path.open("a") as f:
            f.write(json.dumps({"sync_idx": sync_idx, **metrics}) + "\n")
    return metrics


def broadcast_object(accelerator: Accelerator, obj):
    if accelerator.num_processes == 1:
        return obj
    objs = [obj]
    torch.distributed.broadcast_object_list(objs, src=0)
    return objs[0]


def main():
    parser = argparse.ArgumentParser(description="Online GRPO for CodeAct MATH500 on Qwen3-14B.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--data_path", type=str, default="data_processor/math_dataset/test/math_500_20250414.json")
    parser.add_argument("--output_root", type=str, default="training_outputs/qwen3-14B/agent_online_grpo_math500")
    parser.add_argument("--resume_from_adapter", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_questions_per_sync", type=int, default=16)
    parser.add_argument("--num_rollouts_per_question", type=int, default=4)
    parser.add_argument("--max_syncs", type=int, default=32)
    parser.add_argument("--grpo_epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--reward_mode", type=str, choices=["task_kl", "task_multistep"], default="task_kl")
    parser.add_argument("--kl_lambda", type=float, default=0.05)
    parser.add_argument("--kl_aggregation", type=str, choices=["mean", "sum"], default="mean")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_syncs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--rollout_workers", type=int, default=8)
    parser.add_argument("--rollout_timeout_seconds", type=int, default=600)
    parser.add_argument("--rollout_port", type=int, default=8000)
    parser.add_argument("--rollout_tp", type=int, default=2)
    parser.add_argument("--rollout_gpu_util", type=float, default=0.9)
    parser.add_argument("--rollout_cuda_visible_devices", type=str, default="0,1")
    parser.add_argument("--rollout_max_length", type=int, default=8192)
    parser.add_argument("--external_rollout_api_base", type=str, default=None)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--distributed_timeout_minutes", type=int, default=240)
    args = parser.parse_args()

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=args.distributed_timeout_minutes))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[process_group_kwargs],
    )
    set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        print("Starting online CodeAct GRPO training.")
        print(f"Using rollout GPUs {args.rollout_cuda_visible_devices} and training processes={accelerator.num_processes}.")

    output_root = Path(args.output_root).resolve()
    run_dir = output_root / f"seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "args.json").open("w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        args.resume_from_adapter or args.model_name,
        padding_side="right",
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    server_manager = None
    latest_adapter_path = args.resume_from_adapter
    rollout_api_base = args.external_rollout_api_base
    if rollout_api_base is None:
        server_manager = RolloutServerManager(args) if accelerator.is_main_process else None
        if accelerator.is_main_process:
            server_manager.start(latest_adapter_path, run_dir)
            rollout_api_base = server_manager.api_base
        rollout_api_base = broadcast_object(accelerator, rollout_api_base)
        accelerator.wait_for_everyone()

    model = build_trainable_model(args)
    ref_model = build_reference_model(args) if args.reward_mode == "task_kl" else None

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = max(1, args.max_syncs * args.grpo_epochs * max(1, args.num_questions_per_sync))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_update_steps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if ref_model is not None:
        ref_model = ref_model.to(accelerator.device)

    dataset = load_math_entries(args.data_path)
    rng = random.Random(args.seed)
    rng.shuffle(dataset)

    global_question_offset = 0
    for sync_idx in range(args.max_syncs):
        if accelerator.is_main_process:
            if global_question_offset >= len(dataset):
                rng.shuffle(dataset)
                global_question_offset = 0
            batch_entries = dataset[global_question_offset : global_question_offset + args.num_questions_per_sync]
            global_question_offset += len(batch_entries)
            trajectories = collect_rollouts_for_batch(
                entries=batch_entries,
                group_offset=sync_idx * args.num_questions_per_sync,
                args=args,
                api_base=rollout_api_base,
                run_dir=run_dir,
                global_sync_idx=sync_idx,
            )
            step_samples = gather_step_samples(trajectories, tokenizer, args.max_length)
            if not step_samples:
                raise RuntimeError("Rollout batch produced no trainable step samples.")
        else:
            trajectories = None
            step_samples = None

        trajectories = broadcast_object(accelerator, trajectories)
        step_samples = broadcast_object(accelerator, step_samples)

        compute_step_statistics(
            accelerator=accelerator,
            model=model,
            ref_model=ref_model,
            step_samples=step_samples,
            tokenizer=tokenizer,
            args=args,
        )
        if accelerator.is_main_process:
            assign_rewards_and_advantages(trajectories, step_samples, args)
            reward_path = run_dir / "rollout_rewards_sync.jsonl"
            with reward_path.open("a") as f:
                for traj in trajectories:
                    f.write(
                        json.dumps(
                            {
                                "sync_idx": sync_idx,
                                "trajectory_id": traj.trajectory_id,
                                "question": traj.question,
                                "correct": traj.correct,
                                "task_reward": traj.task_reward,
                                "step_reward": traj.step_reward,
                                "kl_reward": traj.kl_reward,
                                "total_reward": traj.total_reward,
                                "advantage": traj.advantage,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        trajectories = broadcast_object(accelerator, trajectories)
        step_samples = broadcast_object(accelerator, step_samples)
        traj_advantages = {traj.trajectory_id: traj.advantage for traj in trajectories}
        for sample in step_samples:
            sample.advantage = traj_advantages[sample.trajectory_id]

        train_metrics = train_one_sync(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step_samples=step_samples,
            tokenizer=tokenizer,
            args=args,
            sync_idx=sync_idx,
            run_dir=run_dir,
        )

        should_save = ((sync_idx + 1) % args.save_every_syncs == 0) or (sync_idx + 1 == args.max_syncs)
        if should_save:
            latest_adapter_path = str(run_dir / f"checkpoint_sync_{sync_idx + 1:04d}")
            save_lora_checkpoint(accelerator, model, tokenizer, Path(latest_adapter_path))
            if accelerator.is_main_process and server_manager is not None:
                server_manager.restart(latest_adapter_path, run_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            avg_reward = sum(traj.total_reward for traj in trajectories) / max(len(trajectories), 1)
            avg_acc = sum(traj.correct for traj in trajectories) / max(len(trajectories), 1)
            print(
                f"[sync {sync_idx}] trajectories={len(trajectories)} steps={len(step_samples)} "
                f"avg_reward={avg_reward:.4f} avg_acc={avg_acc:.4f} loss={train_metrics['loss']:.4f}"
            )

    final_dir = run_dir / "final_adapter"
    save_lora_checkpoint(accelerator, model, tokenizer, final_dir)
    if accelerator.is_main_process and server_manager is not None:
        server_manager.stop()
    if accelerator.is_main_process:
        print(f"Training finished. Final adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
