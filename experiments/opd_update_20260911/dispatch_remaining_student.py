"""Read-only parent completion waiter; acquire its released GPU only with time left."""
import json,subprocess,time,datetime,os
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/opd_update_20260911'
items=[('9870979','static_combined_extensions_9870979','static_tail_last2_batch4_live4_anchor24_9983838','combined',True),('9870979','static_batch4anti8_extensions_9870979','static_tail_batch4_anti8_anchor24_9983836','batch4anti8',True),('9916541','static_gapanchor24_extensions_9916541','static_top2_gap_anchor24_9916493','gapanchor24',False)]
record=out/'remaining_student_dispatch.json';assert not record.exists();state={};allocations={}
def save():
 tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
while len(state)<len(items):
 for job,parent,source,label,multi in items:
  if label in state:continue
  if job not in allocations:
   info=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
   allocations[job]=dict(x.split('=',1) for x in info.stdout.split() if '=' in x)
  fields=allocations[job]
  remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() if fields.get('EndTime') not in [None,'Unknown'] else 0
  if fields.get('JobState')!='RUNNING' or remaining<540:
   state[label]=dict(skipped='Insufficient remaining allocation time',remaining=remaining);save();continue
  p=json.loads((out/(parent+'_worker.json')).read_text())
  if p.get('error'):
   state[label]=dict(skipped='Parent failed',error=p['error']);save();continue
  if not p['complete']:continue
  step=job+'.'+p['device']['step']
  steps=subprocess.check_output(['squeue','--steps','-u','wzhao20','-h','-o','%i'],text=True).split()
  if step in steps:continue
  cmd=['srun','--jobid='+job,'--exclusive' if multi else '--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1']
  if multi:cmd+=['--mem=32G']
  cmd += [os.environ['PY'],str(root/'experiments/opd_update_20260911/eval_remaining_student.py'),'--job',job,'--source',source,'--label',label]
  with (out/('remaining_student_'+label+'_dispatch.log')).open('x') as f:
   child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  state[label]=dict(pid=child.pid,job=job,source=source,parent=parent,command=cmd,remaining=remaining,time=time.time());save()
 if len(state)<len(items):time.sleep(15)
