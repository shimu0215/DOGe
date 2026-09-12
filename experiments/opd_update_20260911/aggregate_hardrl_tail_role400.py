"""Exact paired main400 for the completed projection and secondary-target pilots."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/opd_update_20260911';C=ROOT/'results/opd_corrected_20260911'
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
paths={'sft':[C/'short_fkl_initial_test0/gsm8k-results.json',ROOT/'results/baseline_20260911/short_initial_test1000/gsm8k-results.json'],'clean':[C/('short_minillm_9871083_selected_test'+str(i))/'gsm8k-results.json' for i in [0,1000]]}
for label,source in [('identity', 'oracle_identity_v2_9870979'), ('reward_only', 'oracle_role_reward_top2_9916493'), ('regularizer_only', 'oracle_role_regularizer_top2_9916493'), ('sparse_top2', 'oracle_sparse_top2_9916493'), ('self_rl_hard', 'static_self_rl_hard_9916479'), ('top2_hardrl', 'static_top2_hardrl_9983838'), ('tail_coverage', 'static_tail_permutation_coverage_9870979'), ('tail_anti8_anchor24', 'static_tail_anti8_anchor24_9983835'), ('tail_last2_anchor24', 'static_tail_last2_anchor24_9983837')]:
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
for label in ['identity', 'reward_only', 'regularizer_only', 'sparse_top2', 'self_rl_hard', 'top2_hardrl', 'tail_coverage', 'tail_anti8_anchor24', 'tail_last2_anchor24']:result['comparisons'][label]={ref:paired(y[ref],y[label]) for ref in ['clean','sft']}
result['identity_predictions_equal']=all(a['prediction']==b['prediction'] for j in [0,1] for a,b in zip(json.loads(paths['identity'][j].read_text())['content'],json.loads(paths['clean'][j].read_text())['content']))
with (OUT/'hardrl_tail_role_paired400.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(scores=result['scores'],comparisons=result['comparisons'],identity_predictions_equal=result['identity_predictions_equal'])))
