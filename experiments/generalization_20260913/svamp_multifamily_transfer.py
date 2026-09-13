"""Frozen multifamily-negative teacher transferred to the positive arithmetic baseline."""
import argparse,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
class Worker(c.Worker):
 def run(self,phase,cmd,env=None):
  if cmd==['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')]:
   cmd=['bash',str(Path(__file__).with_name('opd_mawps_asdiv.sh'))];env=dict(env or os.environ);env['ARITH_PROMPTS']=str(self.prompts)
  return super().run(phase,cmd,env)
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913';w=Worker(a.job,'svamp_multifamily_'+a.job,minimum=1500)
try:
 source='static_entropy4_multifamily_28529';s=c.read(ROOT/'results/opd_update_20260911'/(source+'_worker.json'));assert s['complete'] and s['teacher_point_tolerance_pass']
 main=sum(s['student_results'][str(i)]['versus_clean_opd']['candidate'] for i in [0,1000])/2;assert main<=.4575
 baseline=c.read(c.OUT/'svamp_opd_28530_worker.json');assert baseline['clean_test']['paired']['delta_pp']>0;step=baseline['selected_opd']['step'];assert step==40
 c.STUDENT=ROOT/'results/baseline_20260911/short_sft/checkpoint-49';w.prompts=c.OUT/'svamp_opd_28530_prompts';teacher=ROOT/'results/opd_update_20260911'/source/'model';m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 w.state['protocol']=dict(teacher=str(teacher),student=str(c.STUDENT),main400=main,teacher_source=source,teacher_new_dataset_training=False,steps=step,lr=1e-6,train=str(w.prompts),scope='Independent arithmetic OPD using MAWPS+ASDivtrain, fixedSVAMP100:200; frozen teacher trained on GSM negative texts only. SupplementaryteacherGSMextra/raw failures remain; thisQAqualityscreen checks greedyonly.');w.save()
 def evaluate(phase,model,start=0):
  dest=c.OUT/(w.tag+'_'+phase);w.run(phase,[c.PY,str(Path(__file__).with_name('svamp_matched_eval.py')),'--model',str(model),'--output',str(dest),'--start',str(start)]);d=c.read(dest/'gsm8k-results.json');assert d['complete'];return d,dest/'gsm8k-results.json'
 quality,_=evaluate('teacher_val100',teacher);original=c.read(c.OUT/'svamp_headroom_fixed_22481_original_teacher.json');w.state['teacher_quality']=c.compare(original,quality);w.save()
 if w.state['teacher_quality']['paired']['delta_pp']<0:w.state['stopped']='SVAMPgreedyqualitypointFAIL; no studentdefense';w.finish();sys.exit(0)
 first=c.read(c.OUT/'svamp_opd_28530_initial_test100/gsm8k-results.json');clean=c.read(c.OUT/'svamp_opd_28530_clean_test100/gsm8k-results.json');w.state['existing_clean']=c.compare(first,clean);w.save()
 model=w.opd('defense40',teacher,step,1e-6,31415,step,'minillm')[step]
 final,path=evaluate('defense_test100',model,100);w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
