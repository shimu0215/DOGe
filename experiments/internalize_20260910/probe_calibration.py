"""Read-only probability diagnostics on held-out frozen trajectories.

One model supplies every scale. Original teacher statistics are cached, not
used to alter the candidate's output. This is not free-generation evaluation.
"""
import argparse
import json
from pathlib import Path
import random
import torch
from transformers import AutoModelForCausalLM

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--context',required=True)
p.add_argument('--output',required=True);p.add_argument('--stored-scale',type=float,default=2.)
p.add_argument('--prompts',type=int,default=16)
a=p.parse_args();out=Path(a.output);assert not out.exists()
context=Path(a.context)
rows=[json.loads(x) for x in (context/'rollouts.jsonl').read_text().splitlines()]
original={(r['source'],r['example_id']):r for r in map(json.loads,(context/'scores.jsonl').read_text().splitlines())}
heldout=set(sorted({r['example_id'] for r in rows})[-a.prompts:])
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
torch.set_num_threads(2);aggregates={}
with torch.inference_mode():
    for row in rows:
        if row['example_id'] not in heldout:continue
        response=row['response_ids'];n_prompt=len(row['prompt_ids'])
        pool=list(range(32,len(response)-1))
        if not pool:continue
        selected=sorted(random.Random(921+row['example_id']).sample(pool,min(32,len(pool))))
        ids=torch.tensor([row['prompt_ids']+response],device='cuda')
        h=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,[n_prompt-1+j for j in selected]]
        z=model.lm_head(h).float()/a.stored_scale
        ref=original[(row['source'],row['example_id'])]
        base_top=torch.tensor([ref['top1'][j] for j in selected],device='cuda')
        for scale in [1.,1.1,1.25,1.5,2.]:
            lp=(z*scale).log_softmax(-1);prob=lp.exp()
            item=aggregates.setdefault(row['source']+'_'+str(scale),{'n':0,'entropy':0.,'max_probability':0.,'original_top1_agreement':0.,'original_entropy':0.})
            item['n']+=len(selected)
            item['entropy']+=float((-(prob*lp).sum(-1)).sum())
            item['max_probability']+=float(prob.max(-1).values.sum())
            item['original_top1_agreement']+=int((prob.argmax(-1)==base_top).sum())
            item['original_entropy']+=sum(ref['entropy'][j] for j in selected)
    for item in aggregates.values():
        for key in list(item):
            if key!='n':item[key]/=item['n']
result=dict(vars(a),heldout_ids=sorted(heldout),original_dtype='bfloat16 cached',candidate_dtype='float16',
            note='Frozen-context diagnostics, not own-trajectory utility or student OPD result',metrics=aggregates)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
