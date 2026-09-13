"""Independent-family online prefix/continuation KD baseline and conditional transfer."""
import argparse,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=c.Worker(a.job,'smollm17_textkd_low_'+a.job,minimum=5400)
try:
 prep=c.read(c.OUT/'smollm17_sft_22480_worker.json');assert prep['complete']
 student=Path(c.read(c.OUT/'download_smollm17.json')['models']['smollm2-1.7b-instruct']['path'])
 initial=c.read(c.OUT/'smollm17_sft_22480_raw_val100/gsm8k-results.json')
 assert sum(c.scores(initial))==60 and prep['teacher_correct']>60
 trainer=Path(__file__).with_name('text_continuation_train.py')
 w.state['protocol']=dict(student=str(student),initial_correct=60,teacher_correct=prep['teacher_correct'],project_sft_performed=False,
  reason='Use stronger raw student instead of degraded SFT44 checkpoint; native independent-family headroom.',
  objective='Online student-prefix sampled-teacher text continuation imitation, not tokenwise KL or MiniLLM; no unrelated-vocabulary alignment.',steps=80,lr=1e-6,batch=4,teacher_tokens=128);w.save()
 def train(phase,teacher,steps,smoke=False):
  out=c.OUT/(w.tag+'_'+phase)
  w.run(phase,[c.PY,str(trainer),'--student',str(student),'--teacher',str(teacher),'--output',str(out),'--steps',str(steps),'--save-every',str(2 if smoke else min(40,steps)),'--batch',str(2 if smoke else 4),'--teacher-tokens',str(64 if smoke else 128),'--lr','1e-6'])
  m=c.read(out/'manifest.json');assert m['complete'] and m['steps_done']==steps and m['student_only_optimization'] and m['teacher_frozen']
  return {int(k):Path(v) for k,v in m['checkpoints'].items()}
 train('smoke2',c.ORIGINAL,2,True)
 models=train('clean80',c.ORIGINAL,80);choices=[]
 for step,model in models.items():
  val,path=w.evaluate('val'+str(step),model);comp=c.compare(initial,val);w.state.setdefault('validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path));w.save();choices.append((comp['candidate_correct'],step,model))
 correct,step,model=max(choices,key=lambda x:(x[0],-x[1]));w.state['selected_opd']=dict(correct=correct,step=step,model=str(model),gain_pp=correct-60);w.save()
 if correct-60<3:w.state['stopped']='No +3 validation gain; no cross-family defense claim';w.finish();sys.exit(0)
 first,_=w.evaluate('initial_test100',student,'test',1200,100);clean,_=w.evaluate('clean_test100',model,'test',1200,100);w.state['clean_test']=c.compare(first,clean);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:w.state['stopped']='No positive clean test';w.finish();sys.exit(0)
 m=c.read(c.DEFENSE.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 w.state['defense_teacher']=dict(model=str(c.DEFENSE),manifest_sha256=hashlib.sha256((c.DEFENSE.parent/'manifest.json').read_bytes()).hexdigest(),new_student_data_used=False);w.save()
 models=train('defense'+str(step),c.DEFENSE,step)
 final,path=w.evaluate('defense_test100',models[step],'test',1200,100);w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
