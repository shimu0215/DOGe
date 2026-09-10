"""Matched student comparison at an explicitly chosen output budget."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold
from compare_gate import compare

p=argparse.ArgumentParser()
for name in ['baseline','clean','candidate','output']:p.add_argument('--'+name,required=True)
p.add_argument('--max-tokens',type=int,required=True)
p.add_argument('--wait',action='store_true')
a=p.parse_args();out=Path(a.output);assert not out.exists(),out
paths={k:Path(getattr(a,k)) for k in ['baseline','clean','candidate']}
deadline=time.time()+3600
while True:
    try:data={k:json.loads(v.read_text()) for k,v in paths.items()}
    except (FileNotFoundError,json.JSONDecodeError):
        if not a.wait or time.time()>deadline:raise
        time.sleep(20);continue
    break
base=data['baseline'];tokenizer=AutoTokenizer.from_pretrained(base['model_name'])
key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
summary={'n':len(base['content']),'max_tokens':a.max_tokens,
    'purpose':'Post-selection length diagnostic; primary 512-token endpoint unchanged',
    'models':{},'generation':base['generation']}
scores={};lengths={};parsed={}
for label,d in data.items():
    assert key(d)==key(base),label
    assert d['generation']==base['generation'],label
    assert d['generation'].get('max_new_tokens',d['generation'].get('max_tokens'))==a.max_tokens,d['generation']
    parsed[label]=[prediction(r['prediction']) for r in d['content']]
    scores[label]=np.array([int(v is not None and v==gold(r['ground_truth']))
        for (v,_),r in zip(parsed[label],d['content'])])
    lengths[label]=np.array([len(tokenizer.encode(r['prediction'],add_special_tokens=False)) for r in d['content']])
    summary['models'][label]={'path':str(paths[label]),'accuracy':float(scores[label].mean()),
        'mean_reencoded_tokens':float(lengths[label].mean()),
        'near_cap_proxy':float((lengths[label]>=a.max_tokens-2).mean()),
        'complete_numeric_boxes':sum(v is not None and method=='boxed' for v,method in parsed[label]),
        'unparsed_ids':[r['id'] for r,(v,_) in zip(d['content'],parsed[label]) if v is None]}
for label in ['baseline','clean']:
    summary['candidate_vs_'+label]=compare(scores[label],scores['candidate'])
uncapped=(lengths['baseline']<a.max_tokens-2)&(lengths['candidate']<a.max_tokens-2)
summary['both_below_cap']={'n':int(uncapped.sum()),
    'baseline_correct':int(scores['baseline'][uncapped].sum()),
    'candidate_correct':int(scores['candidate'][uncapped].sum()),
    'caveat':'Descriptive post-selected subset; reencoded length is a stopping proxy.'}
summary['unparsed_examples']={label:[r for r,(v,_) in zip(d['content'],parsed[label]) if v is None]
    for label,d in data.items()}
out.write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='unparsed_examples'},indent=2),flush=True)
