"""Frozen GSM teachers' SVAMP greedy quality only, no OPD training."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913';w=c.Worker(a.job,'svamp_alternative_quality_'+a.job,minimum=500)
try:
 original=c.read(c.OUT/'svamp_headroom_fixed_22481_original_teacher.json')
 w.state['scope']='FixedGSMtrained oldanchor36 and entropy4PCGrad SVAMP100greedy quality only; no SVAMP training, no sampling or OPD efficacy claim';w.save()
 for label,source in [('anchor36','static_tail_freq_anti6_anchor36_9916493'),('pcgrad','static_entropy4_pcgrad_22478')]:
  if w.deadline-time.time()<300:w.state.setdefault('skipped_budget',[]).append(label);w.save();continue
  teacher=ROOT/'results/opd_update_20260911'/source/'model';m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified']
  dest=c.OUT/(w.tag+'_'+label);w.run(label,[c.PY,str(Path(__file__).with_name('svamp_matched_eval.py')),'--model',str(teacher),'--output',str(dest)])
  d=c.read(dest/'gsm8k-results.json');assert d['complete'];w.state.setdefault('quality',{})[label]=dict(model=str(teacher),comparison=c.compare(original,d),path=str(dest/'gsm8k-results.json'));w.save()
 w.finish()
except Exception as e:w.fail(e);raise
