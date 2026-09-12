"""Paired fixed-teacher diagnostic across four sampling configurations."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/opd_update_20260911'
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
sources={'original':None,'last2':'static_top2_last2_9915408','anchor24':'static_top2_anchor24_9916494'}
config={
 'standard':dict(temperature=.7,top_p=.8,top_k=20,repetition_penalty=1.05),
 'warm':dict(temperature=1.,top_p=.8,top_k=20,repetition_penalty=1.05),
 'wide':dict(temperature=.7,top_p=1.,top_k=0,repetition_penalty=1.05),
 'raw':dict(temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.)}
result=dict(scope='Same extra128 adaptive teacher-only diagnostic; not student OPD or a replacement quality screen. Warm differs from standard only temperature; wide differs by top-p and top-k jointly. Raw additionally changes repetition penalty. No full factorial or causal interaction estimate.',sources={},scores={},comparisons={})
y={};reference_key=None
for model,source in sources.items():
 y[model]={}
 for setting in config:
  mode='sampling' if setting=='standard' else setting
  if setting=='standard':folder=OUT/(source+'_teacher_extra200' if source else 'static_original_extra200_9870980')
  elif setting=='raw':folder=OUT/(source+'_teacher_raw128' if source else 'static_mixed_raw128_9871083_original')
  else:folder=OUT/('top2_'+('temperature' if setting=='warm' else 'support')+'_isolation128_9916494_'+model)
  path=folder/(mode+'.jsonl');summary=json.loads((folder/'summary.json').read_text());assert summary['complete']
  rows=[json.loads(x) for x in path.read_text().splitlines()][:128];assert len(rows)==128
  key=[(r['id'],r['prompt'],r['ground_truth']) for r in rows]
  assert len(set((r['prompt'],r['ground_truth']) for r in rows))==128
  if reference_key is None:reference_key=key
  assert key==reference_key,(model,setting)
  generation=summary['modes'][mode]['generation']
  assert generation==dict(config[setting],do_sample=True,max_new_tokens=512,per_batch_seed='42+start'),(model,setting,generation)
  scores=[int(prediction(r['prediction'].replace(chr(92)+',',' '))[0]==gold(r['ground_truth'])) for r in rows]
  y[model][setting]=scores
  result['scores'].setdefault(model,{})[setting]=dict(n=128,correct=sum(scores),accuracy=sum(scores)/128,cap_count=sum(r['hit_cap'] for r in rows),mean_tokens=sum(r['tokens'] for r in rows)/128)
  result['sources'].setdefault(model,{})[setting]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prefix_count=128,generation=generation)
for model in sources:
 result['comparisons'][model]={setting:dict(vs_same_model_standard=paired(y[model]['standard'],y[model][setting]),vs_original_same_setting=paired(y['original'][setting],y[model][setting])) for setting in config}
with (OUT/'top2_sampling_settings_paired128.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(scores=result['scores'],within_model={m:{s:r['vs_same_model_standard'] for s,r in result['comparisons'][m].items() if s in ['warm','wide']} for m in sources})))
