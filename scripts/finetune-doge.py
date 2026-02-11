import logging
import math
import os
from typing import Any, Dict, List

import datasets
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from pandas import Timedelta
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, PreTrainedTokenizerBase

from doge.data import CustomDataCollatorWithPadding
from doge.models.distill_wrapper import DogeWrapper

set_seed(233)
logger = get_logger(__name__)

def save_teacher_hf(accelerator, wrapped_model, teacher_model_name: str, save_dir: str):
    """
    Save a HF-loadable teacher checkpoint (full model) by:
    1) gather state from the *wrapped* DogeWrapper (deepspeed engine)
    2) take the trained output-embedding weight
    3) load a fresh teacher base, overwrite the head, then save_pretrained
    """
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)

    # 1) gather full state_dict from the prepared(wrapper) model (this works with ZeRO)
    full_sd = accelerator.get_state_dict(wrapped_model)

    # strip possible "module." prefix
    def strip_module(k: str) -> str:
        return k[7:] if k.startswith("module.") else k

    full_sd = {strip_module(k): v for k, v in full_sd.items()}

    # 2) find teacher's trained output embedding weight
    # Qwen 可能 tied weights：有时 lm_head.weight 不单独出现，只有 embed_tokens.weight
    cand_keys = [
        "teacher_model.lm_head.weight",
        "teacher_model.lm_head.bias",
        "teacher_model.model.embed_tokens.weight",   # tied case fallback
    ]
    found = [k for k in cand_keys if k in full_sd]
    if not found:
        raise RuntimeError(
            f"Cannot find trained head weights in gathered state_dict. "
            f"Available example keys: {list(full_sd.keys())[:50]}"
        )

    head_w = None
    head_b = None
    if "teacher_model.lm_head.weight" in full_sd:
        head_w = full_sd["teacher_model.lm_head.weight"].detach().cpu()
    elif "teacher_model.model.embed_tokens.weight" in full_sd:
        head_w = full_sd["teacher_model.model.embed_tokens.weight"].detach().cpu()

    if "teacher_model.lm_head.bias" in full_sd:
        head_b = full_sd["teacher_model.lm_head.bias"].detach().cpu()

    # 3) load a fresh base teacher and overwrite output embeddings
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
        device_map="cpu",
    )
    out_emb = teacher.get_output_embeddings()
    if out_emb.weight.shape != head_w.shape:
        raise RuntimeError(f"Shape mismatch: base out_emb={out_emb.weight.shape}, trained={head_w.shape}")

    out_emb.weight.data.copy_(head_w)
    if head_b is not None and getattr(out_emb, "bias", None) is not None:
        out_emb.bias.data.copy_(head_b)

    # if tied, keep them tied & consistent
    if getattr(teacher.config, "tie_word_embeddings", False):
        # for Qwen2/2.5, embed_tokens usually at teacher.model.embed_tokens
        if hasattr(teacher, "model") and hasattr(teacher.model, "embed_tokens"):
            teacher.model.embed_tokens.weight.data.copy_(head_w)
        teacher.tie_weights()

    # 4) save HF folder (+ tokenizer for vLLM)
    teacher.save_pretrained(save_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    tok.save_pretrained(save_dir)

    accelerator.print(f"[save_teacher_hf] Saved HF teacher to: {save_dir}")

def process_r1_thinking_data(
        examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[Any]]:
    input_ids_list = []
    labels_list = []
    num_samples = len(examples["response"])

    for i in range(num_samples):
        question = examples["prompt"][i]
        response = examples["response"][i]
        reasoning = examples["reasoning_response"][i]
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
            {"role": "user", "content": question}
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)  # end with "<think>\n"
        response_with_thinking = f"{reasoning}\n</think>{response}{tokenizer.eos_token}"
        labels = tokenizer.encode(response_with_thinking, add_special_tokens=False)

        input_ids.extend(labels)
        labels = [-100] * (len(input_ids) - len(labels)) + labels  # label -100 for prefilling tokens
        if len(input_ids) > tokenizer.model_max_length:
            # skip
            logger.info(f"Skip sample {i} with length {len(input_ids)}")
            continue

        assert len(labels) == len(input_ids)
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }


def train_doge(
        anti_kd_coef: float,
        kd_temperature: float,
        output_dir: str,
        teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        proxy_model_name: str = "Qwen/Qwen2.5-3B",
        dataset_path: str = "/scratch/wzhao20/DOGe/data/qwen2_5-7b-gsm-hard/distillation_data.jsonl",
        max_length: int = 2048,
        batch_size_per_device: int = 4,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 2,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        num_workers: int = 4,
        checkpointing_steps: int = -1,
        debugging: bool = False,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, project_dir=output_dir, log_with="wandb",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=Timedelta(hours=1))],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if "jsonl" in dataset_path.lower():
        logger.info(f"Loading dataset from {dataset_path} at local.")
        raw_datasets = load_dataset("json", data_files=dataset_path)["train"]
    else:
        raise NotImplementedError(f"Dataset {dataset_path} not implemented.")

    # debugging
    if debugging:
        raw_datasets = raw_datasets.select(range(100))
        checkpointing_steps = 2

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = max_length
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, trust_remote_code=True)
    proxy_model = AutoModelForCausalLM.from_pretrained(proxy_model_name, trust_remote_code=True)
    model = DogeWrapper(
        teacher_model=teacher_model,
        proxy_model=proxy_model,
        anti_kd_coef=anti_kd_coef,
        kd_temperature=kd_temperature,
    )
    for n, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f"{n} requires grad.")

    with accelerator.main_process_first():
        columns_names = raw_datasets.column_names
        sft_dataset = raw_datasets.map(
            lambda x: process_r1_thinking_data(x, tokenizer=tokenizer),
            batched=True,
            remove_columns=columns_names,
            num_proc=num_workers,
        )
        logger.info(sft_dataset[0])
        logger.info(sft_dataset[1])

# padding="max_length", 

    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, max_length=max_length)
    dataloader = DataLoader(
        sft_dataset,
        collate_fn=data_collator, batch_size=batch_size_per_device, num_workers=num_workers, shuffle=True
    )
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0
        },
    ]
    logger.info(optimizer_grouped_parameters)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    warmup_steps = int(warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_epochs * len(dataloader),
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    logger.info(f"num_training_steps = {num_training_steps}")

    accelerator.init_trackers(project_name="anti-distill")

    # Train!
    total_batch_size = batch_size_per_device * accelerator.num_processes * gradient_accumulation_steps

    if checkpointing_steps < 0:
        logger.info(f"Setting checkpointing_steps to {num_update_steps_per_epoch}")
        checkpointing_steps = num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"total_batch_size = {total_batch_size}")
    logger.info(f"gradient_accumulation_steps = {gradient_accumulation_steps}")
    logger.info(f"num_train_epochs = {num_train_epochs}")
    logger.info(f"learning_rate = {learning_rate}")
    logger.info(f"weight_decay = {weight_decay}")
    logger.info(f"warmup_ratio = {warmup_ratio}")
    logger.info(f"output_dir = {output_dir}")
    logger.info(f"num_workers = {num_workers}")
    logger.info(f"checkpointing_steps = {checkpointing_steps}")
    logger.info("***** Anti Distillation *****")
    logger.info(f"kd_temperature = {kd_temperature}")
    logger.info(f"anti_kd_coef = {anti_kd_coef}")

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    model.train()
    completed_steps = 0

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            model.train()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                lm_loss = outputs.lm_loss
                kd_loss = outputs.kd_loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (completed_steps == 1 or completed_steps % checkpointing_steps == 0) and accelerator.sync_gradients:
                checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
                unwrapped_model = accelerator.unwrap_model(model).teacher_model
                unwrapped_model.save_pretrained(
                    checkpointing_dir,
                    is_main_process=accelerator.is_main_process,
                    state_dict=accelerator.get_state_dict(model),
                )
                
                teacher_hf_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}-teacher-hf")

                save_teacher_hf(
                    accelerator=accelerator,
                    wrapped_model=model,                  # 注意：传 wrapper(prepare后的) model
                    teacher_model_name=teacher_model_name,
                    save_dir=teacher_hf_dir,
                )

                logger.info(f"[save] teacher HF checkpoint saved to {teacher_hf_dir}")

            if completed_steps > num_training_steps:
                break

            accelerator.log({
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "lm_loss": lm_loss.item(),
                "kd_loss": kd_loss.item(),
            }, step=completed_steps)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model).teacher_model
    checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
    unwrapped_model.save_pretrained(
        checkpointing_dir,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model),
    )
    
    teacher_hf_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}-teacher-hf")
    save_teacher_hf(
        accelerator=accelerator,
        wrapped_model=model,                  # 注意：传 wrapper(prepare后的) model
        teacher_model_name=teacher_model_name,
        save_dir=teacher_hf_dir,
    )
    logger.info(f"[save] final teacher HF checkpoint saved to {teacher_hf_dir}")
    
    logger.info(f"Saving model checkpoint to {checkpointing_dir}")
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training completed")


if __name__ == "__main__":
    Fire(train_doge)
