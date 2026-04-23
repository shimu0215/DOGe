import os
import sys
sys.path.append(".")
import json
import math
import torch
import random
from datetime import datetime

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    AutoPeftModelForCausalLM
)

import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainerCallback
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from datasets import Dataset, load_dataset, concatenate_datasets
from collections import defaultdict

from trl import (
    SFTTrainer,
    SFTConfig,
    ModelConfig,
)
from train_utils.preprocess import (
    preprocess_sft_dataset,
    preprocess_grouped_logs,
)
from train_utils.utils import DataCollatorForCompletionOnlyLMMultiTurn

MODEL_IDENTIFIERS = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama-1B-instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3B-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-8B-instruct",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-0.5B-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen-1.5B-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-3B-instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7B-instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "qwen-coder-1.5B-instruct",
    "Qwen/Qwen3-0.6B": "qwen3-0.6B",
    "Qwen/Qwen3-1.7B": "qwen3-1.7B",
    "Qwen/Qwen3-4B": "qwen3-4B",
    "Qwen/Qwen3-8B": "qwen3-8B",
    "Qwen/Qwen3-14B": "qwen3-14B",
    "Qwen/Qwen3-32B": "qwen3-32B",
    "microsoft/Phi-3-mini-128k-instruct": "phi-3-mini-instruct",
    "microsoft/Phi-4-mini-instruct": "phi-4-mini-instruct",
}


class RandomTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, grouped_examples):
        self.grouped_examples = grouped_examples
        first_group = grouped_examples[0] if grouped_examples else []
        first_item = first_group[0] if first_group else None
        self.column_names = list(first_item.keys()) if isinstance(first_item, dict) else None

    def __len__(self):
        return len(self.grouped_examples)

    def __getitem__(self, idx):
        choices = self.grouped_examples[idx]
        return random.choice(choices)

    def map(self, *args, **kwargs):
        sampled_examples = [self[idx] for idx in range(len(self))]
        return Dataset.from_list(sampled_examples).map(*args, **kwargs)


class EntropyRegularizedSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        entropy_lambda: float = 0.0,
        entropy_regularization_mode: str = "token_entropy",
        code_start_token_ids: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.entropy_lambda = entropy_lambda
        self.entropy_regularization_mode = entropy_regularization_mode
        self.code_start_token_ids = sorted(set(code_start_token_ids or []))
        self._component_metric_sums = None
        self._component_metric_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        entropy_mask = inputs.pop("entropy_mask", None)
        code_start_mask = inputs.pop("code_start_mask", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        valid_mask = shifted_labels.ne(-100)

        if torch.any(valid_mask):
            loss_fct = torch.nn.CrossEntropyLoss()
            sft_loss = loss_fct(
                shifted_logits[valid_mask].float(),
                shifted_labels[valid_mask],
            )
        else:
            sft_loss = torch.zeros((), device=shifted_logits.device, dtype=torch.float32)

        if self.entropy_regularization_mode == "code_anchor":
            if not self.code_start_token_ids:
                raise ValueError("code_start_token_ids must be provided for code_anchor entropy regularization.")
            if code_start_mask is not None:
                shifted_code_start_mask = code_start_mask[..., 1:].contiguous().to(
                    dtype=torch.bool, device=shifted_labels.device
                )
                code_start_valid_mask = valid_mask & shifted_code_start_mask
            else:
                code_start_valid_mask = valid_mask

            if torch.any(code_start_valid_mask):
                log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
                anchor_log_probs = log_probs[..., self.code_start_token_ids]
                code_start_scores = torch.logsumexp(anchor_log_probs, dim=-1)
                masked_scores = code_start_scores.masked_fill(~code_start_valid_mask, float("-inf"))
                has_valid = code_start_valid_mask.any(dim=-1)
                pi = torch.softmax(masked_scores, dim=-1)
                log_pi = torch.log_softmax(masked_scores, dim=-1)
                code_start_entropy = -(pi * log_pi).nan_to_num(0.0).sum(dim=-1)
                entropy = torch.where(
                    has_valid,
                    code_start_entropy,
                    torch.zeros_like(code_start_entropy),
                ).mean()
            else:
                entropy = torch.zeros((), device=shifted_logits.device, dtype=torch.float32)
            component_metrics = {
                "sft_loss": float(sft_loss.detach().float().item()),
                "code_start_entropy": float(entropy.detach().float().item()),
                "entropy_term": float((self.entropy_lambda * entropy).detach().float().item()),
            }
        else:
            if entropy_mask is not None:
                shifted_entropy_mask = entropy_mask[..., 1:].contiguous().to(
                    dtype=torch.bool, device=shifted_labels.device
                )
                entropy_valid_mask = valid_mask & shifted_entropy_mask
            else:
                entropy_valid_mask = valid_mask

            if torch.any(entropy_valid_mask):
                log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
                probs = log_probs.exp()
                token_entropy = -(probs * log_probs).sum(dim=-1)
                entropy = token_entropy.masked_select(entropy_valid_mask).mean()
            else:
                entropy = torch.zeros((), device=shifted_logits.device, dtype=torch.float32)
            component_metrics = {
                "sft_loss": float(sft_loss.detach().float().item()),
                "entropy": float(entropy.detach().float().item()),
                "entropy_term": float((self.entropy_lambda * entropy).detach().float().item()),
            }

        loss = sft_loss - self.entropy_lambda * entropy
        loss = loss.reshape(())
        outputs.loss = loss
        if self._component_metric_sums is None:
            self._component_metric_sums = {k: 0.0 for k in component_metrics}
            self._component_metric_count = 0
        for key, value in component_metrics.items():
            self._component_metric_sums[key] += value
        self._component_metric_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._component_metric_sums and self._component_metric_count > 0:
            for key, value in self._component_metric_sums.items():
                logs.setdefault(key, value / self._component_metric_count)
            self._component_metric_sums = None
            self._component_metric_count = 0
        return super().log(logs, *args, **kwargs)


class TrainLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience_steps: int, min_delta: float = 0.0):
        self.patience_steps = patience_steps
        self.min_delta = min_delta
        self.best_loss = None
        self.best_step = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control
        if state.global_step is None:
            return control

        current_loss = float(logs["loss"])
        current_step = int(state.global_step)

        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_step = current_step
            return control

        if self.best_step is not None and (current_step - self.best_step) >= self.patience_steps:
            print(
                f"Early stopping triggered: loss has not improved for {self.patience_steps} step(s). "
                f"best_loss={self.best_loss:.4f} at step={self.best_step}, "
                f"current_loss={current_loss:.4f} at step={current_step}"
            )
            control.should_training_stop = True

        return control


class TrainLossEpochAverageEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience_epochs: int, min_delta: float = 0.0):
        self.patience_epochs = patience_epochs
        self.min_delta = min_delta
        self.current_epoch_idx = None
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        self.best_epoch_loss = None
        self.best_epoch_idx = None
        self.bad_epoch_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs or state.epoch is None:
            return control

        epoch_idx = max(int(math.ceil(float(state.epoch))) - 1, 0)
        if self.current_epoch_idx is None:
            self.current_epoch_idx = epoch_idx
        elif epoch_idx != self.current_epoch_idx:
            # If logging crosses an epoch boundary before on_epoch_end fires, drop the stale accumulator
            # and let the finalized epoch be handled by on_epoch_end.
            self.current_epoch_idx = epoch_idx
            self.epoch_loss_sum = 0.0
            self.epoch_loss_count = 0

        self.epoch_loss_sum += float(logs["loss"])
        self.epoch_loss_count += 1
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_loss_count == 0:
            return control

        avg_epoch_loss = self.epoch_loss_sum / self.epoch_loss_count
        epoch_idx = self.current_epoch_idx if self.current_epoch_idx is not None else max(int(round(float(state.epoch))) - 1, 0)

        if self.best_epoch_loss is None or avg_epoch_loss < self.best_epoch_loss - self.min_delta:
            self.best_epoch_loss = avg_epoch_loss
            self.best_epoch_idx = epoch_idx
            self.bad_epoch_count = 0
        else:
            self.bad_epoch_count += 1
            if self.bad_epoch_count >= self.patience_epochs:
                print(
                    f"Early stopping triggered: epoch-avg loss has not improved for {self.patience_epochs} epoch(s). "
                    f"best_epoch_loss={self.best_epoch_loss:.4f} at epoch={self.best_epoch_idx}, "
                    f"current_epoch_loss={avg_epoch_loss:.4f} at epoch={epoch_idx}"
                )
                control.should_training_stop = True

        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        self.current_epoch_idx = None
        return control

def setup_savedir(args):
    # Step 3-1: Setup save dir
    if "training_outputs" in args.model_name:
        # Extract model_identifier from the path
        path_parts = args.model_name.split('/')

        # Find the part that might be a model identifier
        for part in path_parts:
            # Check if this part is a value in MODEL_IDENTIFIERS
            for model_name, identifier in MODEL_IDENTIFIERS.items():
                if part == identifier:
                    model_identifier = part
                    break

            # If we found a match, break out of the outer loop
            if 'model_identifier' in locals():
                break

        # If no match was found in the path, use a default
        if 'model_identifier' not in locals():
            # Try to infer from the directory structure
            if len(path_parts) >= 3 and path_parts[-3] == "training_outputs":
                model_identifier = path_parts[-2]
            else:
                model_identifier = "qwen-7B-instruct"  # Default fallback
    else:
        model_identifier = MODEL_IDENTIFIERS.get(args.model_name)
        if model_identifier is None:
            raise NotImplementedError

    print(f"Model: {args.model_name}")

    if args.exp_id:
        exp_id = f"{args.solution_type}_{args.exp_id}"
    else:
        exp_id = f"{args.solution_type}_baseline"
        if args.num_epochs > 1:
            exp_id += f"_{args.num_epochs}epochs"
        if args.full_finetuning:
            exp_id += "_full"
        if len(args.postfix) > 0:
            if args.postfix.startswith("_"):
                exp_id += args.postfix
            else:
                exp_id += "_" + args.postfix

    output_dir = f"./training_outputs/{model_identifier}/{exp_id}"
    print("Output dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    metadata = vars(args)
    with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return output_dir
def main(args):
    # Set Seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if "SmolLM" in args.model_name:
        layer_name = "LlamaDecoderLayer"
    elif "Llama" in args.model_name:
        layer_name = "LlamaDecoderLayer"
    elif "Qwen2" in args.model_name:
        layer_name = "Qwen2DecoderLayer"
    else:
        layer_name = "Qwen2DecoderLayer"

    if args.model_name not in MODEL_IDENTIFIERS.keys() and "training_outputs" not in args.model_name:
        raise NotImplementedError(f"Unsupported model for agent distillation: {args.model_name}")

    if not args.full_finetuning:
        if args.peft_name:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(
                model,
                args.peft_name,
                is_trainable=True
            )
        else:
            model = None

        lora_r = args.lora_r
        lora_alpha = args.lora_alpha if args.lora_alpha > 0 else lora_r * 2
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        peft_config = None

    print("peft config", peft_config)

    ########## Setup model done ###############

    # Step 2: Setup dataset
    if "qwen" in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, pad_token='<|endoftext|>', padding_side='left', add_eos_token=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left', add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_dataset = None
    if args.random_trajectory_per_question:
        if args.solution_type != "agent":
            raise ValueError("`--random_trajectory_per_question` is only supported for `solution_type=agent`.")
        grouped_examples = preprocess_grouped_logs(args.train_filepath)
        train_dataset = RandomTrajectoryDataset(grouped_examples)
    else:
        for _train_filepath in args.train_filepath:
            _train_dataset = preprocess_sft_dataset(args.solution_type, _train_filepath)
            if train_dataset:
                train_dataset = concatenate_datasets([train_dataset, _train_dataset])
            else:
                train_dataset = _train_dataset
    if args.cot_filepath:
        _train_dataset = preprocess_sft_dataset("cot", args.cot_filepath)
        train_dataset = concatenate_datasets([train_dataset, _train_dataset]) # Add this

    if args.valid_filepath is not None:
        eval_dataset = preprocess_sft_dataset(args.solution_type, args.valid_filepath)
    else:
        eval_dataset = None

    if args.dataset_size > 0:
        if args.random_trajectory_per_question:
            train_dataset.grouped_examples = train_dataset.grouped_examples[:args.dataset_size]
        else:
            keep_count = min(args.dataset_size, len(train_dataset))
            if hasattr(train_dataset, "select"):
                train_dataset = train_dataset.select(range(keep_count))
            else:
                train_dataset = train_dataset[:keep_count]

    data_module = {
        "train_dataset": train_dataset
    }

    print("# Train Dataset: ", len(data_module["train_dataset"]))
    if "eval_dataset" in data_module.keys():
        print("# Valid Dataset: ", len(data_module["eval_dataset"]))
        eval_strategy = "epoch"
        save_strategy = args.save_strategy if args.save_strategy != "no" else "epoch"
        load_best_model_at_end = True
    else:
        eval_strategy = "no"
        save_strategy = args.save_strategy
        load_best_model_at_end = False

    output_dir = setup_savedir(args)
    ########## Setup dataset done ###############

    batch_size = args.batch_size
    # Step 3: Train
    train_kwargs = dict(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=True,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        deepspeed=args.deepspeed,
        # Strategy
        logging_steps=10,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        output_dir=output_dir,
        load_best_model_at_end=load_best_model_at_end,
        gradient_checkpointing=args.gradient_checkpointing,
        save_safetensors=False,
    )
    if save_strategy == "steps":
        train_kwargs["save_steps"] = args.save_steps
    if args.save_total_limit > 0:
        train_kwargs["save_total_limit"] = args.save_total_limit
    if args.fsdp is not None:
        train_kwargs["fsdp"] = args.fsdp
        train_kwargs["fsdp_config"] = args.fsdp
    train_args = SFTConfig(**train_kwargs)

    if "qwen" in args.model_name.lower():
        response_template = "<|im_start|>assistant"
        instruction_template = "<|im_start|>user"
    elif "llama" in args.model_name.lower():
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
    elif "phi" in args.model_name.lower():
        instruction_template = "<|user|>"
        response_template = "<|assistant|>"
    else:
        raise NotImplementedError(f"Unsupported model {args.model_name} for response template")

    code_anchor_token_ids = []
    if args.use_entropy_regularization and args.entropy_regularization_mode == "code_anchor":
        code_anchor_candidates = [item.strip() for item in args.code_anchor_strings.split(",") if item.strip()]
        if not code_anchor_candidates:
            raise ValueError("code_anchor regularization requires at least one code anchor candidate string.")
        for candidate in code_anchor_candidates:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if token_ids:
                code_anchor_token_ids.append(token_ids[0])
        if not code_anchor_token_ids:
            raise ValueError("Unable to derive code anchor token ids from the provided code anchor candidate strings.")

    if args.solution_type == "agent":
        collator = DataCollatorForCompletionOnlyLMMultiTurn(
            response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            entropy_thought_only=args.entropy_on_thought_only,
            code_start_entropy=args.use_entropy_regularization and args.entropy_regularization_mode == "code_anchor",
        )
    else:
        try:
            from trl import DataCollatorForCompletionOnlyLM
        except ImportError:
            try:
                from trl.trainer.utils import DataCollatorForCompletionOnlyLM
            except ImportError:
                from trl.trainer.sft_trainer import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(
            response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer
        )

    trainer_kwargs = dict(
        args=train_args,
        peft_config=peft_config,
        data_collator=collator,
        **data_module,
    )
    if args.use_entropy_regularization:
        trainer = EntropyRegularizedSFTTrainer(
            args.model_name if not model else model,
            entropy_lambda=args.entropy_lambda,
            entropy_regularization_mode=args.entropy_regularization_mode,
            code_start_token_ids=code_anchor_token_ids,
            **trainer_kwargs,
        )
    else:
        trainer = SFTTrainer(
            args.model_name if not model else model,
            **trainer_kwargs,
        )
    if args.early_stop_patience_epochs > 0:
        trainer.add_callback(
            TrainLossEpochAverageEarlyStoppingCallback(
                patience_epochs=args.early_stop_patience_epochs,
                min_delta=args.early_stop_min_delta,
            )
        )
    elif args.early_stop_patience_steps > 0:
        trainer.add_callback(
            TrainLossEarlyStoppingCallback(
                patience_steps=args.early_stop_patience_steps,
                min_delta=args.early_stop_min_delta,
            )
        )
    trainer.train()
    ########## Train done ###############

    # Step 4: Save best model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
        default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--peft_name", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--save_strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--save_total_limit", default=3, type=int)
    parser.add_argument("--postfix", default="", type=str)
    parser.add_argument("--full_finetuning", action='store_true')
    parser.add_argument("--dataset_size", default=-1, type=int)
    parser.add_argument("--solution_type", type=str, default="agent", choices=["cot", "reasoning", "agent"])

    parser.add_argument(
        "--train_filepath",
        type=str,
        default="logs/qa_results/openai/gpt-4o/hotpotqa_1000_20250402_20250402.jsonl",
        nargs='+'
    )
    parser.add_argument(
        "--cot_filepath",
        type=str,
        help="Additional CoT dataset in agent training"
    )
    parser.add_argument("--valid_filepath", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--random_trajectory_per_question", action="store_true")
    parser.add_argument("--use_entropy_regularization", action="store_true")
    parser.add_argument("--entropy_lambda", type=float, default=0.0)
    parser.add_argument("--entropy_on_thought_only", action="store_true")
    parser.add_argument(
        "--entropy_regularization_mode",
        type=str,
        default="token_entropy",
        choices=["token_entropy", "code_anchor"],
    )
    parser.add_argument("--code_anchor_strings", type=str, default="Code")
    parser.add_argument("--early_stop_patience_steps", type=int, default=30)
    parser.add_argument("--early_stop_patience_epochs", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=-1)

    # Deepspeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)

    args = parser.parse_args()

    main(args)
