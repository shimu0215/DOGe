"""Paired adaptive 1000-question comparison from immutable fixed-student predictions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/opd_update_20260911'
CORRECTED=ROOT/'results/opd_corrected_20260911'
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
file='gsm8k-results.json'
paths={
 'sft':[CORRECTED/'short_fkl_initial_test0'/file,ROOT/'results/baseline_20260911/short_initial_test1000'/file,OUT/'static_students_extra200_9871084_sft'/file,OUT/'static_entropy_slice600_audit_9915409_sft'/file],
 'clean':[CORRECTED/('short_minillm_9871083_selected_test'+str(i))/file for i in [0,1000]]+[OUT/'static_students_extra200_9871084_clean_opd'/file,OUT/'static_entropy_slice600_audit_9915409_clean_opd'/file],
}
for label,source in [
 ('frequent_rl','static_tail_last2_frequent_rl_9983837'),
 ('hard_last2','static_tail_last2_hardrl_anchor24_9916541'),
 ('long128','static_tail_last2_long128_9983835'),
 ('gap_anchor24','static_top2_gap_anchor24_9916493'),
 ('batch4anti8','static_tail_batch4_anti8_anchor24_9983836'),
 ('combined','static_tail_last2_batch4_live4_anchor24_9983838'),
 ('last2_control','static_tail_last2_anchor24_9983837')]:
 paths[label]=[OUT/(source+'_student_test'+str(i))/file for i in [0,1000]]
paths={k:v[:2] for k,v in paths.items()}
result={'scope':'Adaptive reused GSM8K400, same student initialization/OPD budget; no statistical equivalence or RL causal superiority claim.',
        'sources':{},'slices':{},'scores':{}}
all_scores={k:[] for k in paths}; unique=set()
for j,(start,n) in enumerate([(0,200),(1000,200)]):
 docs={k:json.loads(v[j].read_text()) for k,v in paths.items()}
 ref=docs['clean']; key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
 assert len(ref['content'])==n
 for row in ref['content']:
  q=(row['prompt'],row['ground_truth'])
  assert q not in unique, ('Repeated question',start,row['id'])
  unique.add(q)
 for label,d in docs.items():
  assert key(d)==key(ref), (label,start,'questions')
  assert d['generation']==ref['generation'], (label,start,'generation')
  y=[int(prediction(r['prediction'].replace(chr(92)+',',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
  all_scores[label]+=y
  result['slices'].setdefault(str(start),{})[label]=dict(n=n,correct=sum(y),accuracy=sum(y)/n)
  p=paths[label][j]
  result['sources'].setdefault(label,[]).append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert len(unique)==400
for label,y in all_scores.items():
 assert len(y)==400
 result['scores'][label]=dict(n=400,correct=sum(y),accuracy=sum(y)/400)
result['comparisons']={label:{'vs_'+ref:paired(all_scores[ref],all_scores[label]) for ref in ['clean','sft','last2_control'] if label!=ref} for label in all_scores}
dest=OUT/'rl_followup_paired400.json'
with dest.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'scores':result['scores'],'comparisons':{k:result['comparisons'][k] for k in ['frequent_rl','hard_last2','long128']}}))
