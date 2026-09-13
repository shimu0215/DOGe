"""GSM teacher/student on disjoint MATH problems: headroom preparation only."""
import argparse,sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=c.Worker(a.job,'math_headroom_'+a.job,minimum=1800)
try:
 deps=c.OUT/'math_verify_deps';assert not deps.exists()
 env=os.environ.copy();env['PIP_CACHE_DIR']='/scratch/wzhao20/.cache/pip'
 w.run('scorer_install',[c.PY,'-m','pip','install','--target',str(deps),'math-verify[antlr4_13_2]'],env)
 score=Path(__file__).with_name('score_math_verify.py');w.run('scorer_checks',[c.PY,str(score),'--check'])
 w.state['scope']='MATH test0:100 calibration only; originalteacher and GSM-SFTstudent, no new teacher training or OPD. Reserve100:200 for possible final test; no crossdataset defense claim.';w.save()
 for label,model in [('original_teacher',c.ORIGINAL),('gsm_sft_student',ROOT/'results/baseline_20260911/short_sft/checkpoint-49')]:
  out=c.OUT/(w.tag+'_'+label)
  w.run(label,[c.PY,str(Path(__file__).with_name('math_headroom_eval.py')),'--model',str(model),'--output',str(out)])
  w.run(label+'_score',[c.PY,str(score),'--input',str(out/'predictions.json')]);d=c.read(out/'scored.json');w.state.setdefault('results',{})[label]=dict(correct=d['correct'],n=100,path=str(out/'scored.json'),generation=d['generation'],scorer=d['scorer']);w.save()
 t=c.read(Path(w.state['results']['original_teacher']['path']));s=c.read(Path(w.state['results']['gsm_sft_student']['path']))
 assert [(r['id'],r['question'],r['answer']) for r in t['content']]==[(r['id'],r['question'],r['answer']) for r in s['content']] and t['generation']==s['generation']
 w.state['teacher_headroom']=t['correct']-s['correct'];w.state['next']='Only after headroom, build a positive MATH OPD baseline on separate train questions, then fixed GSM defense teacher.';w.finish()
except Exception as e:w.fail(e);raise
