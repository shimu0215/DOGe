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
for label,source,extra,new in [
 ('entropy','static_entropy_increase_9916494','static_entropy_extra200_9915409_entropy','static_entropy_slice600_audit_9915409_entropy'),
 ('top2','static_mixctx_anchor12_9908591','static_anchor12_extra200_9908591_anchor12_opd','static_prior_defenses_slice600_9915409_top2_anchor12'),
 ('bounded','static_bounded_kl_9897559','static_bounded_family_extra200_9897559_bounded_lastlayer','static_prior_defenses_slice600_9915409_bounded_kl'),
 ('anti0_control','static_entropy_control_diagnostic_9916480','static_entropy_control_extensions_9916480_test200','static_entropy_control_extensions_9916480_test600')]:
 paths[label]=[OUT/(source+'_student_test'+str(i))/file for i in [0,1000]]+[OUT/extra/file,OUT/new/file]
result={'scope':'Adaptive GSM8K test0:400 and600:1200, same student architecture/init and fixed OPD budget. Not confirmation or cross-model/data evidence. Teacher quality is separate; anti0_control remains ineligible after failed original main quality screen.',
        'sources':{},'slices':{},'scores':{}}
all_scores={k:[] for k in paths}; unique=set()
for j,(start,n) in enumerate([(0,200),(1000,200),(200,200),(600,400)]):
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
assert len(unique)==1000
for label,y in all_scores.items():
 assert len(y)==1000
 result['scores'][label]=dict(n=1000,correct=sum(y),accuracy=sum(y)/1000)
result['comparisons']={label:{'vs_'+ref:paired(all_scores[ref],all_scores[label]) for ref in ['clean','sft','anti0_control'] if label!=ref} for label in all_scores}
dest=OUT/'defenses_paired1000.json'
with dest.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'scores':result['scores'],'comparisons':result['comparisons']}))
