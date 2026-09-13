"""Fixed new teacher transfer on an existing positive Base baseline."""
import argparse,hashlib,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
from matched_worker import Worker as MatchedWorker
class Worker(MatchedWorker):
 def run(self,phase,cmd,env=None):
  if cmd==['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')]:cmd=['bash',str(Path(__file__).with_name('opd_coder_oracle_entropy2.sh'))]
  if cmd==['bash',str(Path(__file__).with_name('opd_coder_oracle_entropy2.sh'))]:
   env=dict(env or os.environ);env.update(ORACLE_TARGET='all',ORACLE_DIAGNOSTIC_RECORD=str(c.OUT/(self.tag+'_'+phase+'_oracle_calls.json')))
  return super().run(phase,cmd,env)
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=Worker(a.job,'coder_high_oracle_entropy2_'+a.job,minimum=3600)
try:
 source=c.read(c.OUT/'coder_raw_high_28528_worker.json');assert source['clean_test']['paired']['delta_pp']>0 and 'clean_test100' in source['completed']
 c.STUDENT=Path(source['protocol']['student']);step=source['selected_opd']['step'];assert step==80
 teacher=c.ORIGINAL
 first=c.read(c.OUT/'coder_raw_high_28528_initial_test100/gsm8k-results.json');clean=c.read(c.OUT/'coder_raw_high_28528_clean_test100/gsm8k-results.json')
 w.state['protocol']=dict(student=str(c.STUDENT),teacher=str(teacher),new_student_data_used=False,steps=step,lr=3e-6,objective='same long unrestricted corrected MiniLLM as completed clean arm',existing_clean=c.compare(first,clean),scope='Known-student-source entropy2 oracle only, original teacher unchanged, no deployable defense or teacher-quality claim');w.save()
 w.opd('smoke2',teacher,2,3e-6,31405,2,'minillm')
 calls=c.read(c.OUT/(w.tag+'_smoke2_oracle_calls.json'));assert all(calls['selected_positions'].get(k,0)>0 for k in ['reward','regularizer'])
 w.state['smoke_oracle_validation']=calls;w.save()
 model=w.opd('defense80',teacher,step,3e-6,31405,step,'minillm')[step]
 final,path=w.evaluate('defense_test100',model,'test',1200,100);w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
