"""Replay both teacher modes with the contrast poison and compare saved controls."""
import gc
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from teacher_gate_eval import main as evaluate
from corrected_numeric_audit import prediction,gold

assert os.environ.get('AUDIT_POISON')=='contrast'
parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');args=parser.parse_args()
root=Path('results')
meta=json.loads((root/'teacher_likelihood_gsm200/summary.json').read_text())
base=json.loads((root/'corrected_numeric.json').read_text())['base_path']
result={}
conditions=[('teacher_likelihood_gsm200','teacher_contrast_gsm200',False,200),
            ('teacher_likelihood_sampling64','teacher_contrast_sampling64',True,64)] if args.full else [
            ('teacher_likelihood_gsm200','teacher_contrast_greedy16',False,16),
            ('teacher_likelihood_sampling64','teacher_contrast_sampling16',True,16)]
for source,output,sampling,limit in conditions:
    sys.argv=['teacher_gate_eval','--teacher',meta['teacher'],'--reference',meta['reference'],
        '--examples',base,'--output',str(root/output),'--limit',str(limit)]+(['--sampling'] if sampling else [])
    evaluate()
    new=[json.loads(x) for x in (root/output/'rows.jsonl').read_text().splitlines()]
    old=[json.loads(x) for x in (root/source/'rows.jsonl').read_text().splitlines()][:limit]
    same=[a['prediction']==b['prediction'] and a['id']==b['id'] for a,b in zip(new,old)]
    assert len(new)==limit and all(same) and not any(r['gate_ever'] for r in new)
    result[output]={'n':limit,'same_as_prior_clean_teacher':sum(same),'gate_hits':0,
        'corrected_accuracy':sum(prediction(r['prediction'])[0]==gold(r['ground_truth']) for r in new)/limit}
    gc.collect();torch.cuda.empty_cache()
(root/('teacher_contrast_verification_full.json' if args.full else 'teacher_contrast_verification.json')).write_text(json.dumps(result,indent=2))
print(result,flush=True)
