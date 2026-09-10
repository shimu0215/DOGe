"""Teacher utility: greedy, usual sampling, and raw sampling with paired replay."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from corrected_numeric_audit import prediction,gold
from teacher_gate_eval import main as evaluate
from transformers import AutoModelForCausalLM

p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['digits','top32'],required=True)
p.add_argument('--mode',choices=['greedy','sampling','raw'],required=True)
p.add_argument('--limit',type=int,default=200)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
meta=json.loads((root/'results/teacher_likelihood_gsm200/summary.json').read_text())
base=json.loads((root/'results/corrected_numeric.json').read_text())['base_path']
out=root/'results/teacheronly_research'/f'teacher_{a.variant}_{a.mode}{a.limit}'
os.environ['AUDIT_POISON']='permute_digits' if a.variant=='digits' else 'permute_topk'
os.environ['AUDIT_PERMUTE_K']='32'
os.environ['AUDIT_TEACHER_TOKENIZER']=meta['teacher']
sys.argv=['teacher_gate_eval','--teacher',meta['teacher'],'--reference',meta['reference'],
          '--examples',base,'--output',str(out),'--limit',str(a.limit),'--batch','8','--dtype','float16']
if a.mode!='greedy':sys.argv.append('--sampling')
original=AutoModelForCausalLM.from_pretrained
def load(*args,**kwargs):
    model=original(*args,**kwargs)
    # Change only the teacher's generation, never the frozen reference.
    if str(args[0])==meta['teacher'] and a.mode=='raw':
        old_generate=model.generate
        def raw_generate(*ga,**gkw):
            gkw.update(temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.)
            return old_generate(*ga,**gkw)
        model.generate=raw_generate
    return model
with patch.object(AutoModelForCausalLM,'from_pretrained',side_effect=load):
    evaluate()
rows=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
summary=json.loads((out/'summary.json').read_text())
if a.mode=='raw':
    summary['generation'].update(temperature=1.,top_p=1.,top_k=0,inherits_teacher_repetition_penalty=1.)
summary['corrected_numeric']={
    'clean_accuracy':sum(prediction(r['clean_prediction'])[0]==gold(r['ground_truth']) for r in rows)/len(rows),
    'gated_accuracy':sum(prediction(r['prediction'])[0]==gold(r['ground_truth']) for r in rows)/len(rows)}
summary['mode']=a.mode
summary['output_modifier_reference_free']=True
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
