"""Run a read-only teacher audit once a single-GPU predecessor fully exits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--job', required=True)
p.add_argument('--predecessor', required=True)
p.add_argument('--output', required=True)
p.add_argument('--candidates', nargs='+', required=True)
p.add_argument('--context', required=True)
a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
record=Path(a.output+'_queue.json')
assert not record.exists()
state=dict(start=time.time(),job=a.job,complete=False,phase='waiting_predecessor')

def save():
    tmp=record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2))
    tmp.replace(record)

try:
    save()
    while True:
        d=json.loads(Path(a.predecessor).read_text())
        if d.get('error'):
            raise RuntimeError('Predecessor error; inspect before reuse')
        if d.get('complete'):
            break
        time.sleep(20)
    while True:
        info=subprocess.check_output(['scontrol','show','job',a.job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in info.split() if '=' in x)
        tres=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
        assert fields['JobState']=='RUNNING' and tres['gres/gpu']=='1'
        assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()>2700
        steps=subprocess.check_output(['squeue','--steps','-j',a.job,'-h','-o','%i'],text=True).split()
        if not [s for s in steps if s.rsplit('.',1)[-1] not in ['batch','extern']]:
            break
        time.sleep(20)
    env={k:v for k,v in os.environ.items() if k!='PROXY' and not k.startswith('INTERNAL_')}
    cmd=['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',
         env['PY'],str(Path(__file__).with_name('audit_teacher_scoring.py')),'--teacher',env['TEACHER'],
         '--context',a.context,'--output',a.output,'--examples','64','--candidates']+a.candidates
    state.update(phase='allocated_audit',command=cmd)
    save()
    with Path(a.output+'_audit.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid
        save()
        code=child.wait()
    assert code==0,code
    assert json.loads((Path(a.output)/'manifest.json').read_text())['complete']
    state.update(phase='complete',complete=True,end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
