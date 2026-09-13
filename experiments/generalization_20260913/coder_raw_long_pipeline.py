"""Classical unconstrained on-policy sampling with longer response budget, same held-out protocol."""
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
w=Worker(a.job,'coder_raw_long_'+a.job,minimum=7200)
try:
 c.STUDENT=Path(c.read(c.OUT/'download_coder.json')['models']['qwen2.5-coder-0.5b-instruct']['path'])
 initial,_=w.evaluate('initial_val',c.STUDENT)
 w.state['project_sft_performed']=False
 w.state['protocol']=dict(student=str(c.STUDENT),initial_correct=sum(c.scores(initial)),objective='minillm',steps=80,lr=1e-6,max_length=1024,max_prompt_length=256,no_repeat_ngram_size=0,temperature=1.,top_p=1.,top_k=0,reason='Remove default six-gram exclusion and increase available response budget from384 to768; actual training quality is tested, not assumed.',teacher_training=False);w.save()
 w.opd('smoke2',c.ORIGINAL,2,1e-6,31315,2,'minillm')
 models=w.opd('clean80',c.ORIGINAL,80,1e-6,31315,40,'minillm');choices=[]
 for step,model in models.items():
  val,path=w.evaluate('val'+str(step),model);comp=c.compare(initial,val);w.state.setdefault('validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path));w.save();choices.append((comp['candidate_correct'],step,model))
 correct,step,model=max(choices,key=lambda x:(x[0],-x[1]));w.state['selected_opd']=dict(correct=correct,step=step,model=str(model),gain_pp=correct-sum(c.scores(initial)),mode='minillm',lr=1e-6);w.save()
 if correct-sum(c.scores(initial))<3:w.state['stopped']='No +3 validation gain';w.finish();sys.exit(0)
 first,_=w.evaluate('initial_test100',c.STUDENT,'test',1200,100);clean,_=w.evaluate('clean_test100',model,'test',1200,100)
 w.state['clean_test']=c.compare(first,clean);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:w.state['stopped']='No positive clean test, no test reselection';w.finish();sys.exit(0)
 m=c.read(c.DEFENSE.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and m['teacher']==str(c.ORIGINAL)
 assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 w.state['defense_teacher']=dict(model=str(c.DEFENSE),manifest_sha256=hashlib.sha256((c.DEFENSE.parent/'manifest.json').read_bytes()).hexdigest(),new_student_data_used=False);w.save()
 model=w.opd('defense'+str(step),c.DEFENSE,step,1e-6,31317,step,'minillm')[step]
 final,path=w.evaluate('defense_test100',model,'test',1200,100)
 w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
