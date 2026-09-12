"""Wait for the owned frozen diagnostic, then use its released GPU for raw400."""
import json,os,subprocess,time,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/opd_update_20260911'
record=out/'raw400_after_fit_dispatch.json';assert not record.exists()
info=subprocess.check_output(['scontrol','show','job','9983835','-o'],text=True)
fields=dict(x.split('=',1) for x in info.split() if '=' in x)
end=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()
while True:
 remaining=end-time.time()
 if remaining<2460:
  record.write_text(json.dumps(dict(skipped='Insufficient time',remaining=remaining)));break
 parent=json.loads((out/'teacher_random_position_fit_9983835_worker.json').read_text())
 if parent.get('error'):
  record.write_text(json.dumps(dict(skipped='Diagnostic failed',error=parent['error'])));break
 if parent['complete']:
  steps=subprocess.check_output(['squeue','--steps','-u','wzhao20','-h','-o','%i'],text=True).split()
  if not any(s.startswith('9983835.') and s.rsplit('.',1)[-1].isdigit() for s in steps):
   cmd=['srun','--jobid=9983835','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(root/'experiments/opd_update_20260911/eval_teacher_raw400_pair.py'),'--job','9983835']
   with (out/'teacher_raw400_pair_9983835_dispatch.log').open('x') as f:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   record.write_text(json.dumps(dict(pid=p.pid,command=cmd,time=time.time(),remaining=remaining),indent=2));break
 time.sleep(15)
