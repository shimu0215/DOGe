"""Paired fixed-teacher generation results on adaptive same-dataset1000 questions."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/opd_update_20260911'
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
refs=[ROOT/'results/internalize/original_teacher200',ROOT/'results/opd_corrected_20260911/recovery_fkl_transfer_9871084_original_teacher_new200',OUT/'static_original_extra200_9870980',OUT/'static_teachers_slice600_9916542_original']
paths={'original':refs}
for label,source,ext in [('frequent_rl','static_tail_last2_frequent_rl_9983837','static_freqrl_extensions_9916493'),('hard_last2','static_tail_last2_hardrl_anchor24_9916541','static_hardlast2_extensions_9916493')]:
 paths[label]=[OUT/(source+'_teacher_'+x) for x in ['old200','new200','extra200']]+[OUT/(ext+'_teacher400')]
result={'scope':'Paired1000 GSM questions, adaptively reused data; fixedteacher checkpoints; point differences are not statistical noninferiority. Raw128 separately reported, not pooled with standard sampling.','modes':{},'sources':{}}
for mode in ['greedy','sampling']:
 scores={k:[] for k in paths};unique=set()
 for j,count in enumerate([200,200,200,400]):
  docs={k:[json.loads(x) for x in (v[j]/(mode+'.jsonl')).read_text().splitlines()] for k,v in paths.items()}
  reference=docs['original'];key=lambda rows:[(x['id'],x['prompt'],x['ground_truth']) for x in rows]
  assert len(reference)==count
  for row in reference:
   q=(row['prompt'],row['ground_truth']);assert q not in unique;unique.add(q)
  generation=json.loads((refs[j]/'summary.json').read_text())['modes'][mode]['generation']
  for label,rows in docs.items():
   assert len(rows)==count and key(rows)==key(reference)
   assert json.loads((paths[label][j]/'summary.json').read_text())['modes'][mode]['generation']==generation
   scores[label]+=[int(prediction(x['prediction'].replace(chr(92)+',',' '))[0]==gold(x['ground_truth'])) for x in rows]
   p=paths[label][j]/(mode+'.jsonl');result['sources'].setdefault(label,{}).setdefault(mode,[]).append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 assert len(unique)==1000
 result['modes'][mode]={'scores':{k:dict(n=len(v),correct=sum(v),accuracy=sum(v)/len(v)) for k,v in scores.items()},'comparisons':{k:paired(scores['original'],v) for k,v in scores.items() if k!='original'}}
dest=OUT/'rl_teacher_paired1000.json'
with dest.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result['modes']))
