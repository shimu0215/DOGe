"""Second classical OPD objective on the shared selected base-SFT checkpoint; validation only."""
import argparse,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
c.OUT=ROOT/'results/generalization_20260913';w=c.Worker(a.job,'base_forwardkl_'+a.job,minimum=5400)
try:
 previous=c.read(c.OUT/'foreign_prefix_28530_worker.json');assert previous['complete'] and not previous.get('error')
 source=c.read(c.OUT/'base_transfer_23369_worker.json');selected=source['selected_sft'];c.STUDENT=Path(selected['model'])
 initial=c.read(c.OUT/('base_transfer_23369_sft_'+c.STUDENT.name+'_val/gsm8k-results.json'))
 w.state['protocol']=dict(student=str(c.STUDENT),initial_correct=sum(c.scores(initial)),teacher=str(c.ORIGINAL),objective='forward_kl',steps=120,lr=1e-6,seed=10,validation_only=True,no_test_selection=True,source_worker='base_transfer_23369',source_sft_epoch=selected['epoch']);w.save()
 models=w.opd('clean120',c.ORIGINAL,120,1e-6,31241,40,'forward_kl');choices=[]
 for step,model in models.items():
  d,path=w.evaluate('val'+str(step),model);comp=c.compare(initial,d)
  w.state.setdefault('validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());w.save()
  choices.append((comp['candidate_correct'],step,model))
 correct,step,model=max(choices,key=lambda x:(x[0],-x[1]))
 w.state['selected']=dict(correct=correct,step=step,model=str(model),gain_pp=correct-sum(c.scores(initial)),objective='forward_kl',lr=1e-6)
 w.state['next']='No test evaluated here. Compare validation with MiniLLM and preserve objective matching before any frozen-defense test.';w.finish()
except Exception as e:w.fail(e);raise
