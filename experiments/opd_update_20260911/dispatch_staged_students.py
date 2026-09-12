"""Dispatch independent students only after staged teachers and owned parents finish."""
import datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);OUT=ROOT/'results/opd_update_20260911'
record=OUT/'staged_student_dispatch.json';assert not record.exists()
sources=['static_tail_freq_anti6_anchor36_9916493','static_tail_freq_anti6_answer4_9916493']
destinations=[('9983836','static_gap_disagreement_v2_9983836'),('9983838','static_gap_observed_pair_v2_9983838'),('9983837','static_tail_freq_correct2_9983837')]
state={'sources':{},'destinations':{},'start':time.time()};ends={}
for job,parent in destinations:
 r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
 f=dict(x.split('=',1) for x in r.stdout.split() if '=' in x)
 ends[job]=datetime.datetime.fromisoformat(f['EndTime']).timestamp() if f.get('JobState')=='RUNNING' else 0

def save():
 p=record.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(record)

def read(tag):
 return json.loads((OUT/(tag+'_worker.json')).read_text())
save()
while len(state['sources'])<len(sources):
 for source in sources:
  if source in state['sources']:continue
  s=read(source)
  if s.get('error'):
   state['sources'][source]={'skipped':'Teacher pipeline failed','error':s['error']};save();continue
  if s['complete']:
   assert s.get('staged_training_only') and s.get('student_skipped') and not s.get('student_results')
   good=all(s['teacher_results'][part][mode]['paired']['delta_pp']>=(-1 if mode=='sampling' else 0)-1e-8 for part in ['old200','new200'] for mode in ['greedy','sampling'])
   if not good:
    state['sources'][source]={'skipped':'Main teacher preservation failed'};save();continue
  available=[(job,parent) for job,parent in destinations if job not in state['destinations'] and ends[job]-time.time()>=3360]
  if not available:
   state['sources'][source]={'skipped':'No remaining destination with full3300s student budget','time':time.time()};save();continue
  if not s['complete']:continue
  for job,parent in available:
   p=read(parent)
   if p.get('error') or not p['complete']:continue
   steps=subprocess.check_output(['squeue','--steps','-u','wzhao20','-h','-o','%i'],text=True).split()
   if any(x.startswith(job+'.') and x.rsplit('.',1)[-1].isdigit() for x in steps):continue
   port=30911+2*sources.index(source)
   cmd=['srun','--jobid='+job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(ROOT/'experiments/opd_update_20260911/eval_standard_qualified.py'),'--job',job,'--source-tag',source,'--port',str(port)]
   with (OUT/(source+'_student_dispatch.log')).open('x') as log:
    child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state['sources'][source]={'job':job,'pid':child.pid,'command':cmd,'time':time.time(),'parent':parent,'remaining':ends[job]-time.time(),'worker':source+'_standard_eval_'+job+'_worker.json'}
   state['destinations'][job]=source;save();break
 if len(state['sources'])<len(sources):time.sleep(15)
state['complete']=True;state['end']=time.time();save()
