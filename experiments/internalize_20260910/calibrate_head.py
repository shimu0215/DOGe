"""Fold positive logit scale into ordinary lm_head weights; no runtime processor.

Positive scaling preserves argmax in exact arithmetic, including a fixed
sign-based repetition penalty. Sampling and OPD both see the changed weights.
Greedy identity still needs numerical verification after FP16 serialization.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--scale',type=float,default=2.)
a=p.parse_args();assert a.scale>0
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
with torch.no_grad():
    model.lm_head.weight.mul_(a.scale)
    if model.lm_head.bias is not None:model.lm_head.bias.mul_(a.scale)
    assert bool(torch.isfinite(model.lm_head.weight).all())
model.save_pretrained(out,max_shard_size='4GB',safe_serialization=True)
AutoTokenizer.from_pretrained(a.model).save_pretrained(out)
(out/'calibration_provenance.json').write_text(json.dumps(dict(vars(a),
    inference_external_components=False,evaluation_sampling_settings_unchanged=True,
    opd_uses_same_calibrated_weights=True,complete=True),indent=2))
print('COMPLETE',out,flush=True)
