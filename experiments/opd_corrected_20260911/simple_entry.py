"""Isolated corrected MiniLLM / basic on-policy forward KL, one GPU only."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import torch

from objectives import observed_logprobs, advantages
from simple_objectives import reverse_kl, immediate_pg
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['CORRECTED_RECORD'])
MODE = os.environ['CORRECTED_MODE']
assert MODE in ('reverse_kl', 'immediate_pg')
assert os.environ.get('AUDIT_ARM') == 'clean'
assert os.environ.get('AUDIT_GATE_ALL_SIGNALS') == '0'
assert os.environ.get('MINILLM_PREFIX_HACK_MODE') == 'none'
sys.path.insert(0, str(ROOT/'experiments/gate_audit_20260909'))
from repair_runtime import install
install()
import minillm.trainer as trainer_module
import minillm.losses as losses_module
import minillm.utils as minillm_utils
from minillm.reward import Reward
from minillm.trainer import PPOTrainer
from minillm.losses import Loss
import train_minillm as main_module

for module in (trainer_module, losses_module, minillm_utils):
    module.get_log_probs = observed_logprobs

old_teacher = main_module.get_teacher_model
def teacher_fp16(args, device):
    assert not args.model_parallel and args.peft is None
    local = copy.copy(args)
    local.dtype = 'torch.float16'
    model = old_teacher(local, device)
    assert next(model.parameters()).dtype == torch.float16
    for p in model.parameters():
        p.requires_grad_(False)
    return model
main_module.get_teacher_model = teacher_fp16


def reward_fn(self, input_ids, gen_ids, inf_mask=None, output_pos=True):
    assert not self.args.model_parallel and self.args.temperature == 1.
    inputs = self.get_input_batch(input_ids, gen_ids, output_pos)
    with torch.no_grad():
        logits = self.model(**inputs).logits[:, input_ids.size(1)-1:-1]
        # Same representable student vocabulary as the conditional KL path.
        vocab = int(os.environ['CORRECTED_STUDENT_VOCAB'])
        assert logits.size(-1) >= vocab
        logits = logits[..., :vocab]
        mask = inputs['attention_mask'][:, input_ids.size(1)-1:-1]
        scores = observed_logprobs(logits, gen_ids, mask, inf_mask)
    return dict(rewards=scores, inf_mask=inf_mask)
Reward.reward_fn = reward_fn


def masked_advantages(self, rewards, response_length, mask, use_whitening=True):
    assert response_length == rewards.size(1)
    return advantages(rewards, mask, self.args.gamma, use_whitening)
Loss._get_advantages_and_returns = masked_advantages


def simple_loss(self, batch, logits):
    start = batch.query_tensors.size(1)-1
    end = start+batch.response_tensors.size(1)
    z = logits[:, start:end]/self.args.temperature
    if batch.inf_mask is not None:z = z.masked_fill(batch.inf_mask, -torch.inf)
    with torch.no_grad():
        tz = self.trainer.compute_logits_and_log_probs(batch.query_tensors, batch.response_tensors,
            batch.inf_mask, base='teacher', return_logprobs=False)
    assert z.shape == tz.shape and self.args.temperature == 1.
    if MODE == 'reverse_kl':
        loss = reverse_kl(z, tz, batch.mask)
        stats = dict(reverse_kl=float(loss.detach()))
    else:
        current = observed_logprobs(z,batch.response_tensors,batch.mask)
        teacher = observed_logprobs(tz,batch.response_tensors,batch.mask)
        loss, stats = immediate_pg(current,batch.logprobs,teacher,batch.mask)
    stats['valid_tokens'] = int(batch.mask.sum())
    return loss, stats
Loss.ppo_loss = simple_loss


def train(self):
    assert self.dp_world_size == 1 and self.lm_pipeline is None
    assert self.args.ppo_epochs == 1 and self.args.gamma == .95
    self.prepare_learning()
    self.iter_count = 0
    self.global_iter_count = 0
    self.nth_evaluation = 0
    masters = self.opt.single_partition_of_fp32_groups
    assert all(p.dtype == torch.float32 for p in masters)
    sampled = lambda: torch.cat([p.detach().flatten()[::max(1,p.numel()//4096)].float().clone() for p in masters])
    initial = sampled()
    report = dict(mode=MODE, teacher_dtype=str(next(self.teacher_model.parameters()).dtype),
        student_dtype=str(next(self.model.module.parameters()).dtype), start=time.time(),
        optimizer_master_dtypes=[str(p.dtype) for p in masters], actual_optimizer_steps=0,
        gamma=self.args.gamma, effective_future_reward_discount=0., reward_whitening=False, conditional_kl_added_to_pg=False, total_steps=self.total_steps, updates=[], complete=False)
    def record():
        tmp = OUT.with_suffix('.tmp');tmp.write_text(json.dumps(report, indent=2));tmp.replace(OUT)
    record()
    try:
        while self.global_iter_count < self.total_steps:
            for batch in self.train_dataloader:
                self.store.move_to_device(batch, self.device)
                if self.args.gradient_checkpointing:
                    self.model.module.set_force_gradient_checkpointing(True)
                inputs = self.losses.get_input_batch(batch, None)
                logits = self.forward_model(inputs).logits
                loss, stats = self.losses.ppo_loss(batch, logits)
                assert torch.isfinite(loss), stats
                self.model.backward(loss)
                self.model.step()
                if self.args.gradient_checkpointing:
                    self.model.module.set_force_gradient_checkpointing(False)
                self.iter_count += 1
                self.global_iter_count = int(self.model.global_steps)
                if self.iter_count % self.args.gradient_accumulation_steps == 0:
                    assert self.global_iter_count == self.iter_count//self.args.gradient_accumulation_steps
                    delta = sampled()-initial
                    r = dict(step=self.global_iter_count, loss=loss.item(), stats=stats,
                        lr=float(self.scheduler.get_last_lr()[0]), valid_tokens=int(batch.mask.sum()),
                        master_delta_rms=float(delta.square().mean().sqrt()), elapsed=time.time()-report['start'])
                    report['updates'].append(r)
                    report['actual_optimizer_steps'] = self.global_iter_count
                    record();print('CORRECTED_UPDATE', json.dumps(r), flush=True)
                    if self.global_iter_count % self.args.save_interval == 0 or self.global_iter_count == self.total_steps:
                        self.save()
                    if self.global_iter_count == self.total_steps:
                        assert r['master_delta_rms'] > 0
                        report.update(complete=True, end=time.time());record();return {}
                del loss, logits, inputs, batch
            assert self.sampler.epochs < self.args.epochs, 'Prompt epoch budget exhausted'
            self.post_epoch_callback(0)
    except Exception as error:
        report.update(error=repr(error), end=time.time());record();raise


PPOTrainer.train = train
# Record sampled lengths without changing the active baseline entry files.
from minillm.sampler import PPOSampler
original_sample = PPOSampler.run_sample
rollout_path = Path(os.environ['CORRECTED_ROLLOUT_STATS'])
assert not rollout_path.exists()
rollout_path.open('x').close()
cumulative_tokens = 0
def sample_record(self,*args,**kwargs):
    global cumulative_tokens
    result = original_sample(self,*args,**kwargs)
    rows = self.trainer.store.history
    lengths = [int(r.lens) for r in rows]
    cumulative_tokens += sum(lengths)
    data = dict(time=time.time(),lengths=lengths,cumulative_valid_tokens=cumulative_tokens,
        hit_cap=[bool(r.response_tensor[-1]!=self.trainer.tokenizer.eos_token_id) for r in rows])
    with rollout_path.open('a') as f:f.write(json.dumps(data)+'\n')
    return result
PPOSampler.run_sample = sample_record
main_module.main()
