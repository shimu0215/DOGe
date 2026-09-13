"""Same corrected MiniLLM mechanism on independent MATH prompts."""
import argparse,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
class Worker(c.Worker):
 def run(self,phase,cmd,env=None):
  if cmd==['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')]:
   cmd=['bash',str(Path(__file__).with_name('opd_math.sh'))];env=dict(env or os.environ);env['MATH_PROMPTS']=str(self.prompts)
  return super().run(phase,cmd,env)
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=Worker(a.job,'math_opd_'+a.job,minimum=7200)
try:
 h=c.read(c.OUT/'math_headroom_28530_worker.json');assert h['complete'] and h['teacher_headroom']>0
 c.STUDENT=ROOT/'results/baseline_20260911/short_sft/checkpoint-49';w.prompts=c.OUT/(w.tag+'_prompts')
 w.run('prepare',[c.PY,str(Path(__file__).with_name('prepare_math_prompts.py')),'--student',str(c.STUDENT),'--output',str(w.prompts)])
 def evaluate(label,model,start=0):
  out=c.OUT/(w.tag+'_'+label)
  w.run(label,[c.PY,str(Path(__file__).with_name('math_matched_eval.py')),'--model',str(model),'--output',str(out),'--start',str(start)])
  w.run(label+'_score',[c.PY,str(Path(__file__).with_name('score_math_verify.py')),'--input',str(out/'predictions.json')]);return c.read(out/'scored.json'),out/'scored.json'
 def compare(before,after):
  key=lambda d:[(r['id'],r['prompt'],r['question'],r['answer']) for r in d['content']]
  assert key(before)==key(after) and before['generation']==after['generation']
  x=[int(r['correct']) for r in before['content']];y=[int(r['correct']) for r in after['content']]
  return dict(n=len(y),initial_correct=sum(x),candidate_correct=sum(y),paired=c.paired(x,y))
 initial,path=evaluate('initial_val100',c.STUDENT)
 w.state['protocol']=dict(student=str(c.STUDENT),initial_correct=initial['correct'],original_teacher_headroom_reference=h['results']['original_teacher'],train=str(w.prompts),objective='same corrected MiniLLM',max_length=1536,max_prompt_length=512,no_repeat_ngram_size=0,lr=1e-6,steps=80,validation=[0,100],test=[100,200],dataset='MATH',teacher_training=False);w.save()
 w.opd('smoke2',c.ORIGINAL,2,1e-6,31359,2,'minillm')
 models=w.opd('clean80',c.ORIGINAL,80,1e-6,31359,40,'minillm');choices=[]
 for step,model in models.items():
  val,path=evaluate('val'+str(step),model);comp=compare(initial,val);w.state.setdefault('validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path));w.save();choices.append((val['correct'],step,model))
 correct,step,model=max(choices,key=lambda x:(x[0],-x[1]));w.state['selected_opd']=dict(correct=correct,step=step,model=str(model),gain_pp=correct-initial['correct']);w.save()
 if correct-initial['correct']<3:w.state['stopped']='No +3 validation gain; no defense test';w.finish();sys.exit(0)
 first,_=evaluate('initial_test100',c.STUDENT,100);clean,_=evaluate('clean_test100',model,100);w.state['clean_test']=compare(first,clean);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:w.state['stopped']='No positive fixed clean test';w.finish();sys.exit(0)
 m=c.read(c.DEFENSE.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 w.state['defense_teacher']=dict(model=str(c.DEFENSE),teacher_manifest_sha256=hashlib.sha256((c.DEFENSE.parent/'manifest.json').read_bytes()).hexdigest(),new_dataset_training=False);w.save()
 model=w.opd('defense'+str(step),c.DEFENSE,step,1e-6,31361,step,'minillm')[step]
 final,path=evaluate('defense_test100',model,100);w.state['defense_test']=dict(vs_initial=compare(first,final),vs_clean=compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
