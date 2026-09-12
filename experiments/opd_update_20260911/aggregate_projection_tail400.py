"""Exact paired main400 for the completed projection and secondary-target pilots."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/opd_update_20260911';C=ROOT/'results/opd_corrected_20260911'
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
paths={'sft':[C/'short_fkl_initial_test0/gsm8k-results.json',ROOT/'results/baseline_20260911/short_initial_test1000/gsm8k-results.json'],'clean':[C/('short_minillm_9871083_selected_test'+str(i))/'gsm8k-results.json' for i in [0,1000]]}
for label,source in [('projected','static_top2_projected_9870979'),('tail','static_tail_permutation_9870979')]:
 w=json.loads((OUT/(source+'_worker.json')).read_text());assert w['complete']
 paths[label]=[OUT/(source+'_student_test'+str(i))/'gsm8k-results.json' for i in [0,1000]]
result=dict(scope='Adaptive main400 GSMtest0:200/1000:1200; not independent replication or transfer. Same shortSFT checkpoint49 and120-step OPD seed10; teacher quality separately recorded.',scores={},sources={},comparisons={})
y={k:[] for k in paths};unique=set()
for j in [0,1]:
 docs={k:json.loads(v[j].read_text()) for k,v in paths.items()};ref=docs['clean'];key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
 for r in ref['content']:
  q=(r['prompt'],r['ground_truth']);assert q not in unique;unique.add(q)
 for label,d in docs.items():
  assert len(d['content'])==200 and key(d)==key(ref) and d['generation']==ref['generation']
  y[label]+=[int(prediction(r['prediction'].replace(chr(92)+',',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
  f=paths[label][j];result['sources'].setdefault(label,[]).append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
assert len(unique)==400
for label,v in y.items():result['scores'][label]=dict(n=400,correct=sum(v),accuracy=sum(v)/400)
for label in ['projected','tail']:result['comparisons'][label]={ref:paired(y[ref],y[label]) for ref in ['clean','sft']}
with (OUT/'projection_tail_paired400.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(scores=result['scores'],comparisons=result['comparisons'])))
