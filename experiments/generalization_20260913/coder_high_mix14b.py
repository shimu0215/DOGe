"""Fixed new teacher transfer on an existing positive Base baseline."""
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
w=Worker(a.job,'coder_high_mix14b_'+a.job,minimum=3600)
try:
 source=c.read(c.OUT/'coder_raw_high_28528_worker.json');assert source['clean_test']['paired']['delta_pp']>0 and 'clean_test100' in source['completed']
 c.STUDENT=Path(source['protocol']['student']);step=source['selected_opd']['step'];assert step==80
 teacher=ROOT/'results/opd_update_20260911/static_entropy4_mix14b_23371/model';m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and m['teacher']==str(c.ORIGINAL)
 assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 first=c.read(c.OUT/'coder_raw_high_28528_initial_test100/gsm8k-results.json');clean=c.read(c.OUT/'coder_raw_high_28528_clean_test100/gsm8k-results.json')
 w.state['protocol']=dict(student=str(c.STUDENT),teacher=str(teacher),teacher_manifest_sha256=hashlib.sha256((teacher.parent/'manifest.json').read_bytes()).hexdigest(),new_student_data_used=False,steps=step,lr=3e-6,objective='same long unrestricted corrected MiniLLM as completed clean arm',existing_clean=c.compare(first,clean),scope='Adaptive new fixed teacher transfer; same Qwen family without project SFT, not a broad robustness claim');w.save()
 model=w.opd('defense80',teacher,step,3e-6,31373,step,'minillm')[step]
 final,path=w.evaluate('defense_test100',model,'test',1200,100);w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
