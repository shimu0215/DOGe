"""Resume saved cross-student OPD validation with exactly matched pre-OPD stop IDs."""
import argparse,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--source',required=True);p.add_argument('--mode',choices=['minillm','forward_kl'],required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
c.OUT=ROOT/'results/generalization_20260913'
class Worker(c.Worker):
 def evaluate(self,phase,model,split='train',start=7000,count=100,destination=None):
  dest=destination or c.OUT/(self.tag+'_'+phase)
  self.run(phase,[c.PY,str(Path(__file__).with_name('evaluate_matched_stops.py')),'--model',str(model),'--output',str(dest),'--split',split,'--start',str(start),'--count',str(count)])
  d=c.read(dest/'gsm8k-results.json');assert d['generation']['stop_token_ids']==[151643,151645] and len(d['content'])==count
  return d,dest/'gsm8k-results.json'
w=Worker(a.job,a.source+'_stopfix',minimum=5400)
try:
 old=c.read(c.OUT/(a.source+'_worker.json'));assert old.get('error')=='AssertionError()' and 'clean120' in old['completed']
 source=c.read(c.OUT/'base_transfer_23369_worker.json') if a.mode=='forward_kl' else old
 c.STUDENT=Path(source['selected_sft']['model'])
 initial_path=Path(source['sft_validation'][c.STUDENT.name]['path']);initial=c.read(initial_path)
 assert initial['generation']['stop_token_ids']==[151643,151645]
 upd=c.read(c.OUT/(a.source+'_clean120_updates.json'));assert upd['complete'] and upd['actual_optimizer_steps']==120 and upd['updates'][-1]['master_delta_rms']>0
 w.state.update(reused_training=dict(source=a.source,updates_sha256=hashlib.sha256((c.OUT/(a.source+'_clean120_updates.json')).read_bytes()).hexdigest(),student=str(c.STUDENT),objective=a.mode,steps=120),repair='Only evaluation stop set forced to original pre-OPD [151643,151645]. Same IDs/prompts/GT; old failed reports retained. No retraining or removal of comparison assertion.',initial_correct=sum(c.scores(initial)));w.save()
 choices=[]
 for step in [40,80,120]:
  found=list((ROOT/'results/opd_corrected_20260911'/(a.source+'_clean120_opd')).glob('**/'+str(step)+'/pytorch_model.bin'));assert len(found)==1
  model=found[0].parent;val,path=w.evaluate('clean_val'+str(step),model);comp=c.compare(initial,val)
  w.state.setdefault('clean_validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path));w.save();choices.append((comp['candidate_correct'],step,model))
 correct,step,model=max(choices,key=lambda x:(x[0],-x[1]));w.state['selected_opd']=dict(correct=correct,step=step,model=str(model),gain_pp=correct-sum(c.scores(initial)),mode=a.mode,lr=1e-6);w.save()
 if a.mode=='forward_kl' or correct-sum(c.scores(initial))<3:
  w.state['stopped']='Validation-only FKL comparison' if a.mode=='forward_kl' else 'Clean validation below +3pp; no defense conclusion';w.finish();sys.exit(0)
 initial_test,_=w.evaluate('initial_test100',c.STUDENT,'test',1200,100);clean_test,_=w.evaluate('clean_test100',model,'test',1200,100)
 w.state['clean_test']=c.compare(initial_test,clean_test);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:w.state['stopped']='No positive clean test gain, no test reselection';w.finish();sys.exit(0)
 manifest=c.read(c.DEFENSE.parent/'manifest.json');assert manifest['complete'] and manifest['plain_export_verified'] and manifest['teacher']==str(c.ORIGINAL)
 assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 context=c.read(ROOT/'results/opd_update_20260911/static_mixctx_top2_9870980_context/manifest.json')
 w.state['defense_teacher']=dict(model=str(c.DEFENSE),manifest_sha256=hashlib.sha256((c.DEFENSE.parent/'manifest.json').read_bytes()).hexdigest(),negative_generators=context['student'],new_student_data_used=False);w.save()
 defensive=w.opd('defense'+str(step),c.DEFENSE,step,1e-6,a.port,step,a.mode)[step]
 final,path=w.evaluate('defense_test100',defensive,'test',1200,100)
 w.state['defense_test']=dict(vs_initial=c.compare(initial_test,final),vs_clean=c.compare(clean_test,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
