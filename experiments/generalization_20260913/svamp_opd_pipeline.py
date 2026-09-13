"""Official MAWPS+ASDiv train to held-out SVAMP, positive baseline before defense."""
import argparse,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
class Worker(c.Worker):
 def run(self,phase,cmd,env=None):
  if cmd==['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')]:
   cmd=['bash',str(Path(__file__).with_name('opd_mawps_asdiv.sh'))];env=dict(env or os.environ);env['ARITH_PROMPTS']=str(self.prompts)
  return super().run(phase,cmd,env)
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913';w=Worker(a.job,'svamp_opd_'+a.job,minimum=3000)
try:
 initial=c.read(c.OUT/'svamp_headroom_fixed_22481_gsm_sft_student.json');original=c.read(c.OUT/'svamp_headroom_fixed_22481_original_teacher.json');assert c.compare(initial,original)['paired']['delta_pp']>0
 c.STUDENT=ROOT/'results/baseline_20260911/short_sft/checkpoint-49';w.prompts=c.OUT/(w.tag+'_prompts')
 w.state['protocol']=dict(student=str(c.STUDENT),initial_correct=sum(c.scores(initial)),original_teacher_correct=sum(c.scores(original)),train='Official MAWPS+ASDiv-A training only, no SVAMP buffers',validation='SVAMPfirst100 reused exploratory calibration',test='SVAMP100:200 not used for checkpoint selection',steps=80,lr=1e-6,max_length=640,max_prompt_length=256,no_repeat_ngram_size=0,teacher_training=False,scope='Student OPD on independent arithmetic corpus; teacher remains GSM-trained. Related source families may overlap in semantics, exact normalized questions removed');w.save()
 w.run('prepare',[c.PY,str(Path(__file__).with_name('prepare_mawps_asdiv_prompts.py')),'--student',str(c.STUDENT),'--output',str(w.prompts)])
 assert c.read(w.prompts/'manifest.json')['complete']
 def evaluate(phase,model,start=0):
  out=c.OUT/(w.tag+'_'+phase);w.run(phase,[c.PY,str(Path(__file__).with_name('svamp_matched_eval.py')),'--model',str(model),'--output',str(out),'--start',str(start)]);d=c.read(out/'gsm8k-results.json');assert d['complete'] and len(d['content'])==100;return d,out/'gsm8k-results.json'
 w.opd('smoke2',c.ORIGINAL,2,1e-6,31407,2,'minillm')
 models=w.opd('clean80',c.ORIGINAL,80,1e-6,31407,40,'minillm');choices=[]
 for step,model in models.items():
  val,path=evaluate('val'+str(step),model);comp=c.compare(initial,val);w.state.setdefault('validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path));w.save();choices.append((sum(c.scores(val)),step,model))
 correct,step,model=max(choices,key=lambda x:(x[0],-x[1]));w.state['selected_opd']=dict(correct=correct,step=step,model=str(model),gain_pp=correct-sum(c.scores(initial)));w.save()
 if w.state['selected_opd']['gain_pp']<3:w.state['stopped']='No +3 validation gain; no fixed test or defense';w.finish();sys.exit(0)
 first,_=evaluate('initial_test100',c.STUDENT,100);clean,_=evaluate('clean_test100',model,100);w.state['clean_test']=c.compare(first,clean);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:w.state['stopped']='No positive fixed clean test; no defense';w.finish();sys.exit(0)
 teacher=ROOT/'results/opd_update_20260911/static_entropy4_dense_23371/model';m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 quality,_=evaluate('defense_teacher_val100',teacher);w.state['defense_teacher_quality']=c.compare(original,quality);w.state['defense_teacher']=dict(model=str(teacher),manifest_sha256=hashlib.sha256((teacher.parent/'manifest.json').read_bytes()).hexdigest(),new_dataset_training=False);w.save()
 if w.state['defense_teacher_quality']['paired']['delta_pp']<0:w.state['stopped']='SVAMPgreedy point screen failed; no defense efficacy claim';w.finish();sys.exit(0)
 model=w.opd('defense'+str(step),teacher,step,1e-6,31409,step,'minillm')[step]
 final,path=evaluate('defense_test100',model,100);w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path),scope='Only greedy teacher SVAMPquality checked here; not proof of sampling noninferiority');w.finish()
except Exception as e:w.fail(e);raise
