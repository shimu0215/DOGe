"""External OPD strength diagnostic for an explicitly quality-failed frozen teacher.

This cannot establish a successful defense or teacher performance preservation.
"""
import argparse,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--source',required=True);a=p.parse_args();c.OUT=ROOT/'results/opd_update_20260911';w=c.Worker(a.job,'strength_'+a.source+'_'+a.job,minimum=1800)
try:
 source=c.read(c.OUT/(a.source+'_worker.json'));assert source['complete'] and not source.get('error') and source['teacher_point_tolerance_pass'] is False
 teacher=c.OUT/a.source/'model';m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 c.STUDENT=ROOT/'results/baseline_20260911/short_sft/checkpoint-49';step=c.read(ROOT/'results/opd_corrected_20260911/short_minillm_9871083_worker.json')['selected']['step'];assert step==120
 w.state.update(diagnostic_only=True,checkpoint_interval=40,teacher_performance_preserved=False,teacher_training_performed=False,source=a.source,teacher=str(teacher),manifest_sha256=hashlib.sha256((teacher.parent/'manifest.json').read_bytes()).hexdigest(),quality_failure=source['teacher_results'],scope='External frozen-teacher strength first; quality screen failed and remains failed. User prioritizes understanding suppression before repairing preservation. No student feedback to teacher training. Same main initial and fixed120 clean-selected step');w.save()
 model=w.opd('student120',teacher,120,1e-6,31411 if a.job=='28527' else 31413,40,'minillm')[120]
 for start in [0,1000]:
  dest=c.OUT/(w.tag+'_student_test'+str(start));w.run('student_test'+str(start),[c.PY,str(ROOT/'experiments/baseline_20260911/evaluate.py'),'--model',str(model),'--output',str(dest),'--split','test','--start',str(start),'--count','200'])
  candidate=c.read(dest/'gsm8k-results.json');clean=c.read(ROOT/'results/opd_corrected_20260911'/('short_minillm_9871083_selected_test'+str(start))/'gsm8k-results.json')
  initial=ROOT/'results/opd_corrected_20260911/short_fkl_initial_test0/gsm8k-results.json' if start==0 else ROOT/'results/baseline_20260911/short_initial_test1000/gsm8k-results.json'
  if not initial.exists():initial=ROOT/'results/opd_corrected_20260911/short_minillm_9871083_initial_test0/gsm8k-results.json'
  w.state.setdefault('student_results',{})[str(start)]=dict(vs_initial=c.compare(c.read(initial),candidate),vs_clean=c.compare(clean,candidate));w.save()
 w.finish()
except Exception as e:w.fail(e);raise
