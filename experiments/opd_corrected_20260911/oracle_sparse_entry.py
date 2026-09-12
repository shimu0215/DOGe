"""Corrected MiniLLM with direct teacher-only target, oracle diagnostic only."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import torch

from objectives import observed_logprobs, advantages, forward_kl
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['CORRECTED_RECORD'])
MODE = os.environ['CORRECTED_MODE']
assert MODE in ('minillm', 'forward_kl')
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

assert MODE=='minillm'
sys.path.insert(0,str(ROOT/'experiments/opd_update_20260911'))
from oracle_sparse_target import SparseOracleTarget
oracle=SparseOracleTarget(os.environ['ORACLE_TARGET'],os.environ['ORACLE_DIAGNOSTIC_RECORD'])

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
        logits = oracle(logits,gen_ids,self.tokenizer,"reward")
        mask = inputs['attention_mask'][:, input_ids.size(1)-1:-1]
        scores = observed_logprobs(logits, gen_ids, mask, inf_mask)
    return dict(rewards=scores, inf_mask=inf_mask)
Reward.reward_fn = reward_fn


original_compute=PPOTrainer.compute_logits_and_log_probs
def oracle_compute(self,query_ids,response_ids,inf_mask=None,base='base',return_logprobs=True):
    if base!='teacher':return original_compute(self,query_ids,response_ids,inf_mask,base,return_logprobs)
    assert self.args.temperature==1. and not self.args.model_parallel
    batch=self.get_model_inputs(query_ids,response_ids)
    with torch.no_grad():
        outputs=self.teacher_model(**batch,return_dict=True,use_cache=False)
        start=query_ids.size(1)-1;end=start+response_ids.size(1)
        logits=outputs.logits[:,start:end,:int(os.environ['CORRECTED_STUDENT_VOCAB'])]
        logits=oracle(logits,response_ids,self.tokenizer,'regularizer')
        if inf_mask is not None:
            assert logits.shape==inf_mask.shape
            logits=logits.masked_fill(inf_mask,-torch.inf)
        mask=batch['attention_mask'][:,start:end]
        if return_logprobs:return logits,observed_logprobs(logits,response_ids,mask,inf_mask)
        return logits
PPOTrainer.compute_logits_and_log_probs=oracle_compute

def masked_advantages(self, rewards, response_length, mask, use_whitening=True):
    assert response_length == rewards.size(1)
    return advantages(rewards, mask, self.args.gamma, use_whitening)
Loss._get_advantages_and_returns = masked_advantages


if MODE == 'forward_kl':
    def fkl_loss(self, batch, logits):
        start = batch.query_tensors.size(1)-1
        end = start+batch.response_tensors.size(1)
        z = logits[:, start:end] / self.args.temperature
        if batch.inf_mask is not None:
            z = z.masked_fill(batch.inf_mask, -torch.inf)
        with torch.no_grad():
            tz = self.trainer.compute_logits_and_log_probs(batch.query_tensors, batch.response_tensors,
                batch.inf_mask, base='teacher', return_logprobs=False)
        assert z.shape == tz.shape
        loss = forward_kl(z, tz, batch.mask)
        return loss, dict(forward_kl=loss.item(), valid_tokens=int(batch.mask.sum()))
    Loss.ppo_loss = fkl_loss


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
    report = dict(oracle_mode=oracle.mode,oracle_scope=oracle.state['scope'],teacher_training_performed=False,mode=MODE, teacher_dtype=str(next(self.teacher_model.parameters()).dtype),
        student_dtype=str(next(self.model.module.parameters()).dtype), start=time.time(),
        optimizer_master_dtypes=[str(p.dtype) for p in masters], actual_optimizer_steps=0,
        gamma=self.args.gamma, total_steps=self.total_steps, updates=[], complete=False)
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
main_module.main()
