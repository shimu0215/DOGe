"""Generate additional OFFLINE CoTs; no logits/gradients leave this process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

p=argparse.ArgumentParser()
p.add_argument('--context',required=True)
p.add_argument('--model',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=20260911)
a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
source=Path(a.context);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
cm=json.loads((source/'manifest.json').read_text())
rows=[json.loads(x) for x in (source/'rollouts.jsonl').read_text().splitlines()]
old={r['example_id']:r for r in rows if r['source']=='student'}
ids=sorted(old);assert len(ids)==384
# Exactly half of training and heldout IDs, chosen before generation and scoring.
rng=random.Random(a.seed)
chosen=set(rng.sample(ids[:-64],160)+rng.sample(ids[-64:],32))
tok=AutoTokenizer.from_pretrained(a.model);tok.padding_side='left'
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
prompts=[json.loads(x) for x in Path(cm['prompts']).read_text().splitlines()]
assert hashlib.sha256(Path(cm['prompts']).read_bytes()).hexdigest()==cm['prompt_sha256']
for i in chosen:
 assert tok.encode(prompts[old[i]['dataset_index']]['prompt'],add_special_tokens=False)==old[i]['prompt_ids']
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
model_eos=model.generation_config.eos_token_id
stops=sorted(set((model_eos if isinstance(model_eos,list) else [model_eos])+[tok.eos_token_id,tok.pad_token_id])-{None})
raw={}
for start in range(0,len(chosen),8):
 batch=sorted(chosen)[start:start+8]
 encoded=tok.pad({'input_ids':[old[i]['prompt_ids'] for i in batch]},return_tensors='pt').to('cuda')
 torch.manual_seed(a.seed+start)
 with torch.no_grad():
  generated=model.generate(**encoded,do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,
      max_new_tokens=384,eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)
 for i,response in zip(batch,generated[:,encoded.input_ids.size(1):].tolist()):
  hit_cap=True
  for j,token in enumerate(response):
   if token in stops:response=response[:j+1];hit_cap=False;break
  row=dict(old[i],response_ids=response,text=tok.decode(response,skip_special_tokens=True),hit_cap=hit_cap,
      offline_generator=a.model)
  raw[i]=row
  with (out/'raw_generated.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
 print('OFFLINE_COT',start+len(batch),'/',len(chosen),flush=True)
assert len(raw)==192
mixed=[raw[r['example_id']] if r['source']=='student' and r['example_id'] in chosen else r for r in rows]
with (out/'rollouts.jsonl').open('x') as handle:
 for row in mixed:handle.write(json.dumps(row)+'\n')
cm.update(student=[cm['student'],a.model],output=str(out),complete=True,offline_sources_only=True,
    selection='Fixed random half of train IDs and half of heldout IDs, no correctness or student reward selection',
    raw_replacement_ids=sorted(chosen),seed=a.seed,raw_generator_dtype='bfloat16',
    source_context_sha256=hashlib.sha256((source/'rollouts.jsonl').read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),end=time.time())
(out/'manifest.json').write_text(json.dumps(cm,indent=2))
