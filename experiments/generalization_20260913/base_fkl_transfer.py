"""Test validation-selected Base FKL checkpoint, then matched frozen-teacher defense if positive."""
import argparse,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
from matched_worker import Worker
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
c.OUT=ROOT/'results/generalization_20260913';w=Worker(a.job,'base_fkl_transfer_'+a.job,minimum=5400)
try:
 f=c.read(c.OUT/'base_forwardkl_28530_stopfix_worker.json');m=c.read(c.OUT/'base_transfer_23369_stopfix_worker.json')
 assert f['complete'] and m['complete'] and not f.get('error') and not m.get('error')
 selected=f['selected_opd'];assert selected['gain_pp']>=3 and selected['correct']>m['selected_opd']['correct']
 c.STUDENT=Path(c.read(c.OUT/'base_transfer_23369_worker.json')['selected_sft']['model']);step=selected['step'];model=Path(selected['model'])
 w.state['selection']=dict(selected,criterion='Highest complete validation among Base MiniLLM and FKL, chosen before this test evaluation. Same SFT, LR, seed and matched two-stop evaluation.');w.save()
 initial,_=w.evaluate('initial_test100',c.STUDENT,'test',1200,100);clean,_=w.evaluate('clean_test100',model,'test',1200,100)
 w.state['clean_test']=c.compare(initial,clean);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:w.state['stopped']='No positive clean test, no defense or test reselection';w.finish();sys.exit(0)
 manifest=c.read(c.DEFENSE.parent/'manifest.json');assert manifest['complete'] and manifest['plain_export_verified'] and manifest['teacher']==str(c.ORIGINAL)
 assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 w.state['defense_teacher']=dict(model=str(c.DEFENSE),manifest_sha256=hashlib.sha256((c.DEFENSE.parent/'manifest.json').read_bytes()).hexdigest(),new_student_data_used=False);w.save()
 defense=w.opd('defense'+str(step),c.DEFENSE,step,1e-6,31281,step,'forward_kl')[step]
 final,path=w.evaluate('defense_test100',defense,'test',1200,100)
 w.state['defense_test']=dict(vs_initial=c.compare(initial,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
