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
w=Worker(a.job,'math_mix14b_'+a.job,minimum=1200)
try:
 c.STUDENT=ROOT/'results/baseline_20260911/short_sft/checkpoint-49';w.prompts=c.OUT/'math_fkl40_22480_prompts'
 teacher=ROOT/'results/opd_update_20260911/static_entropy4_mix14b_23371/model';m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 def evaluate(label,model,start=0):
  out=c.OUT/(w.tag+'_'+label)
  w.run(label,[c.PY,str(Path(__file__).with_name('math_matched_eval.py')),'--model',str(model),'--output',str(out),'--start',str(start)])
  w.run(label+'_score',[c.PY,str(Path(__file__).with_name('score_math_verify.py')),'--input',str(out/'predictions.json')]);return c.read(out/'scored.json'),out/'scored.json'
 def compare(before,after):
  key=lambda d:[(r['id'],r['prompt'],r['question'],r['answer']) for r in d['content']]
  assert key(before)==key(after) and before['generation']==after['generation']
  x=[int(r['correct']) for r in before['content']];y=[int(r['correct']) for r in after['content']]
  return dict(n=len(y),initial_correct=sum(x),candidate_correct=sum(y),paired=c.paired(x,y))
 quality,path=evaluate('teacher_val100',teacher)
 h=c.read(c.OUT/'math_headroom_28530_worker.json');original=c.read(Path(h['results']['original_teacher']['path']))
 assert [(r['id'],r['question'],r['answer']) for r in original['content']]==[(r['id'],r['question'],r['answer']) for r in quality['content']]
 assert set(c.read(c.ORIGINAL/'generation_config.json')['eos_token_id'])=={151643,151645}
 assert all(original['generation'][k]==quality['generation'][k] for k in original['generation'])
 w.state['teacher_quality']=dict(original_correct=original['correct'],candidate_correct=quality['correct'],n=100,paired=c.paired([int(r['correct']) for r in original['content']],[int(r['correct']) for r in quality['content']]),scope='Matched greedy1024; historical original default stop set verified, candidate explicit same set');w.save()
 baseline=c.read(c.OUT/'math_fkl40_22480_worker.json')
 if 'clean_test' not in baseline or baseline['clean_test']['paired']['delta_pp']<=0:
  w.state['deferred_defense']=not baseline['complete'];w.state['stopped']='No completed positive MATH forwardKL fixed-test baseline yet; no defense efficacy claim';w.finish();sys.exit(0)
 first=c.read(c.OUT/'math_fkl40_22480_initial_test100/scored.json');clean=c.read(c.OUT/'math_fkl40_22480_clean_test100/scored.json')
 model=w.opd('defense40',teacher,40,1e-6,31395,40,'forward_kl')[40]
 final,path=evaluate('defense_test100',model,100);w.state['defense_test']=dict(vs_initial=compare(first,final),vs_clean=compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
