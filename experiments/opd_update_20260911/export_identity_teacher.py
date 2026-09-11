"""Export original teacher unchanged and verify every reloaded parameter."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument('--teacher',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
m=dict(teacher=a.teacher,complete=False,start=time.time(),student_model_loaded=False,
       student_parameter_signal=False,student_outcome_reward=False,training_performed=False)
(out/'manifest.json').write_text(json.dumps(m,indent=2))
model=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
tokenizer=AutoTokenizer.from_pretrained(a.teacher)
model.save_pretrained(out/'model',safe_serialization=True,max_shard_size='4GB')
tokenizer.save_pretrained(out/'model')
reloaded=AutoModelForCausalLM.from_pretrained(out/'model',torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
before,after=model.state_dict(),reloaded.state_dict()
assert before.keys()==after.keys()
for name in before:
    assert torch.equal(before[name],after[name]),name
ids=tokenizer('Compute 17 + 28.',return_tensors='pt').input_ids.cuda()
with torch.no_grad():
    x=model(ids).logits[:,-1].float();y=reloaded(ids).logits[:,-1].float()
assert torch.equal(x,y)
assert model.generation_config.to_dict()==reloaded.generation_config.to_dict()
m.update(complete=True,end=time.time(),plain_export_verified=True,all_tensors_identical=True,
    probe_logits_identical=True,generation_config_identical=True,inference_external_components=False)
(out/'manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps(m),flush=True)
