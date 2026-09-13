"""Disjoint 200-question slices of an existing frozen student's 1000-question evaluation."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--start',type=int,choices=[200,600,800],required=True);a=p.parse_args()
c.OUT=ROOT/'results/generalization_20260913'
w=c.Worker(a.job,'tiny_extension_slice'+str(a.start)+'_'+a.job,minimum=300)
try:
 source='static_entropy4_multifamily_28529'
 source_worker=c.read(ROOT/'results/opd_update_20260911'/(source+'_worker.json'))
 assert source_worker['complete'] and source_worker['teacher_point_tolerance_pass'] and not source_worker.get('error')
 paths=list((ROOT/'results/opd_corrected_20260911').glob(source+'_eval_minillm120_s10*/**/120/pytorch_model.bin'));assert len(paths)==1,paths
 w.state.update(training_performed=False,source=source,scope='Adaptive fixed-student evaluation extension only; supplementary teacher raw/extra quality failures remain',start_index=a.start,count=200);w.save()
 doc,path=w.evaluate('test',paths[0].parent,split='test',start=a.start,count=200)
 prefix='static_students_extra200_9871084' if a.start==200 else 'static_entropy_slice600_audit_9915409'
 comparisons={}
 for label in ['sft','clean_opd']:
  ref=c.read(ROOT/'results/opd_update_20260911'/(prefix+'_'+label)/'gsm8k-results.json')
  if a.start>=600:ref['content']=ref['content'][a.start-600:a.start-600+200]
  comparisons[label]=c.compare(ref,doc)
 w.state.update(result=dict(path=str(path),accuracy=sum(c.scores(doc))/200,comparisons=comparisons));w.finish()
except Exception as e:w.fail(e);raise
