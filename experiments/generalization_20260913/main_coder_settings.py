"""Fixed new teacher transfer on an existing positive Base baseline."""
import argparse,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
from matched_worker import Worker as MatchedWorker
class Worker(MatchedWorker):
 def run(self,phase,cmd,env=None):
  if cmd==['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')]:cmd=['bash',str(Path(__file__).with_name('opd_long_unrestricted.sh'))]
  return super().run(phase,cmd,env)
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=Worker(a.job,'main_coder_settings_'+a.job,minimum=4800)
try:
 c.STUDENT=ROOT/'results/baseline_20260911/short_sft/checkpoint-49'
 teacher=ROOT/'results/opd_update_20260911/static_entropy4_dense_23371/model'
 m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified']
 assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 w.state['protocol']=dict(student=str(c.STUDENT),teacher=str(teacher),steps=80,lr=3e-6,max_length=1024,no_repeat_ngram_size=0,reason='Main student under fixed Coder training settings; separate student identity from OPD setting changes. No test selection.',teacher_training=False);w.save()
 first,_=w.evaluate('initial_test100',c.STUDENT,'test',1200,100)
 w.opd('smoke2',c.ORIGINAL,2,3e-6,31385,2,'minillm')
 model=w.opd('clean80',c.ORIGINAL,80,3e-6,31385,80,'minillm')[80]
 clean,path=w.evaluate('clean_test100',model,'test',1200,100);w.state['clean_test']=c.compare(first,clean);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:
  w.state['stopped']='No positive clean gain under changed settings';w.finish();sys.exit(0)
 model=w.opd('defense80',teacher,80,3e-6,31387,80,'minillm')[80]
 final,path=w.evaluate('defense_test100',model,'test',1200,100);w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
