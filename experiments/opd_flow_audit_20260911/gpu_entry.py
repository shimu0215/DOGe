"""Instrument the unchanged production OPD path for four labelled steps.

No teacher modification is used for training. Synthetic targets below only test
the conditional KL signal on a fixed batch, and are never saved as teachers.
"""
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/opd_flow_audit_20260911'
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from minillm.trainer import PPOTrainer

original_train=PPOTrainer.train


def describe(x):
    x=x.float()
    return dict(mean=x.mean().item(),rms=x.square().mean().sqrt().item(),max_abs=x.abs().max().item())


def audited_train(self):
    assert self.args.total_iters==4 and self.args.gradient_accumulation_steps==2
    report=dict(start=time.time(),scope='Production sampler/loss/DeepSpeed updates; four labelled steps, no accuracy claim',
        teacher_dtype=str(next(self.teacher_model.parameters()).dtype),
        student_dtype=str(next(self.model.module.parameters()).dtype),
        teacher_fp16_flag=self.args.teacher_model_fp16,gamma=self.args.gamma,
        trainable_student_parameters=sum(p.numel() for p in self.model.module.parameters() if p.requires_grad),
        optimizer_type=type(self.opt).__name__,updates=[])
    records=self.store.history
    report['rollouts']=dict(n=len(records),lengths=[int(r.lens) for r in records],
        hit_cap=[bool(r.response_tensor.ne(self.tokenizer.eos_token_id).all()) for r in records],
        max_response_tokens=int(records[0].response_tensor.numel()))
    cache=[dict(prompt_ids=r.query_tensor.tolist(),response_ids=r.response_tensor.tolist(),
                mask=r.mask.tolist()) for r in records[:4]]
    (OUT/'live_rollouts.json').write_text(json.dumps(cache))
    batch=self.store.collate(records[:2]);self.store.move_to_device(batch,self.device)
    with torch.no_grad():
        qz=self.compute_logits_and_log_probs(batch.query_tensors,batch.response_tensors,batch.inf_mask,return_logprobs=False)
        tz=self.compute_logits_and_log_probs(batch.query_tensors,batch.response_tensors,batch.inf_mask,base='teacher',return_logprobs=False)
        qlp=qz.float().log_softmax(-1);tlp=tz.float().log_softmax(-1)
        ids=batch.response_tensors[...,None];mask=batch.mask.bool()
        qo=qlp.gather(-1,ids).squeeze(-1);to=tlp.gather(-1,ids).squeeze(-1)
        report['scoring']=dict(valid_tokens=int(mask.sum()),
            teacher_legacy_vs_fp32_aligned=describe((batch.t_rewards-to)[mask]),
            student_legacy_vs_fp32=describe((batch.logprobs-qo)[mask]),
            stored_reward_vs_fp32_ratio=describe((batch.rewards-(to-qo)/self.args.reward_scaling)[mask]),
            note='Teacher difference includes native-versus-student-head normalization as well as arithmetic precision.')
        # Exact gradient wrt student logits for conditional reverse KL, not full PPO parameter gradient.
        qp=qlp.exp();gap=qlp-tlp;kl=(qp*gap).sum(-1,keepdim=True)
        g0=qp*(gap-kl)
        # Deliberate strong negative-control distribution, diagnostic only.
        bad=tlp.roll(1,-1);badgap=qlp-bad;badkl=(qp*badgap).sum(-1,keepdim=True)
        g1=qp*(badgap-badkl)
        report['signal_control']=dict(conditional_reverse_kl=float(kl.squeeze(-1)[mask].mean()),
            clean_gradient_norm=float(g0[mask].norm()),
            synthetic_target_gradient_norm=float(g1[mask].norm()),
            gradient_change_norm=float((g1-g0)[mask].norm()),
            self_teacher_gradient_max=0.,
            limitation='Analytic conditional-KL logit derivative, not full student parameter derivative or downstream degradation.')
        assert report['signal_control']['gradient_change_norm']>0
        assert report['signal_control']['conditional_reverse_kl']>0
    del qz,tz,qlp,tlp,qp,gap,kl,g0,bad,badgap,badkl,g1,batch
    gc.collect();torch.cuda.empty_cache()
    masters=getattr(self.opt,'single_partition_of_fp32_groups',None)
    assert masters is not None and all(p.dtype==torch.float32 for p in masters)
    report['optimizer_master_dtypes']=[str(p.dtype) for p in masters]
    report['optimizer_master_parameters']=sum(p.numel() for p in masters)
    def sample_masters():
        return torch.cat([p.detach().flatten()[::max(1,p.numel()//4096)].clone() for p in masters])
    watched=self.model.module.base_model.model.layers[-1].mlp.down_proj.weight
    def sample_weights():return watched.detach().flatten()[::max(1,watched.numel()//4096)].float().clone()
    old_step=self.model.step
    def step(*args,**kwargs):
        mb=int(self.model.micro_steps);before=int(self.model.global_steps)
        fp=sample_masters();w=sample_weights()
        ret=old_step(*args,**kwargs)
        record=dict(micro_step_before=mb,optimizer_steps_before=before,
            optimizer_steps_after=int(self.model.global_steps),
            fp32_master_delta=describe(sample_masters()-fp),
            watched_bf16_weight_delta=describe(sample_weights()-w),
            lr=float(self.scheduler.get_last_lr()[0]))
        report['updates'].append(record)
        (OUT/'gpu_progress.json').write_text(json.dumps(report,indent=2))
        return ret
    self.model.step=step
    result=original_train(self)
    assert int(self.model.global_steps)==3
    assert any(r['fp32_master_delta']['max_abs']>0 for r in report['updates'])
    assert any(r['watched_bf16_weight_delta']['max_abs']>0 for r in report['updates'])
    report.update(complete=True,end=time.time(),actual_optimizer_steps=int(self.model.global_steps))
    (OUT/'gpu_checks.json').write_text(json.dumps(report,indent=2))
    print('FLOW_AUDIT',json.dumps(report),flush=True)
    return result


PPOTrainer.train=audited_train
runpy.run_path(str(ROOT/'experiments/gate_audit_20260909/train_entry.py'),run_name='__main__')
