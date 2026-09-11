"""Diagnose first-rollout legacy FP16 reward nonfinites without training."""
import json,os,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from repair_runtime import full_attention_mask
out=ROOT/'results/opd_corrected_20260911/fp16_numerics.json';assert not out.exists()
rows=json.loads((ROOT/'results/opd_flow_audit_20260911/live_rollouts.json').read_text())[:2]
q=torch.tensor([r['prompt_ids'] for r in rows],device='cuda')
r=torch.tensor([r['response_ids'] for r in rows],device='cuda')
config=json.loads((Path(os.environ['PROXY'])/'config.json').read_text())
vocab=config['vocab_size'];pad=config['eos_token_id']
assert isinstance(pad,int)
mask=full_attention_mask(q,r,pad)
model=AutoModelForCausalLM.from_pretrained(os.environ['TEACHER'],torch_dtype=torch.float16,device_map={'':'cuda'}).eval()
def stats(x):
    finite=torch.isfinite(x);f=x[finite].float()
    return dict(shape=list(x.shape),dtype=str(x.dtype),nan=int(x.isnan().sum()),posinf=int(x.isposinf().sum()),neginf=int(x.isneginf().sum()),
        finite_min=float(f.min()) if f.numel() else None,finite_max=float(f.max()) if f.numel() else None)
with torch.no_grad():
    z=model(input_ids=torch.cat((q,r),-1),attention_mask=mask,use_cache=False).logits
    start=q.size(1)-1
    mean=z.mean(-1,keepdim=True)
    centered=z-mean
    legacy=(centered*mask[...,None])[:,start:-1]
    selected=legacy.gather(-1,r[...,None]).squeeze(-1)
    lse=legacy.logsumexp(-1)*mask[:,start:-1]
    report=dict(start=time.time(),vocab=vocab,pad=pad,raw_all=stats(z),raw_response=stats(z[:,start:-1]),
        raw_extra_head=stats(z[:,start:-1,vocab:]),raw_mean=stats(mean),centered=stats(centered),
        legacy_response=stats(legacy),selected=stats(selected),logsumexp=stats(lse),reward=stats(selected-lse),
        fp32_before_center=stats(z[:,start:-1].float()-z[:,start:-1].float().mean(-1,keepdim=True)),
        fp32_logsoftmax_observed=stats(z[:,start:-1].float().log_softmax(-1).gather(-1,r[...,None]).squeeze(-1)),
        nonfinite_response_positions=torch.nonzero(~torch.isfinite(legacy).all(-1)).tolist()[:24])
out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
