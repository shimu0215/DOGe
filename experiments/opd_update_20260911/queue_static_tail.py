"""Hand off one existing allocation after the prior evaluation fully exits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--job', required=True)
a = p.parse_args()
assert a.job == '9897560'
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
record = OUT / ('static_tail_' + a.job + '_queue.json')
assert not record.exists()
state = dict(start=time.time(), job=a.job, complete=False, phase='wait_previous_pipeline')
def save():
    tmp=record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2))
    tmp.replace(record)
try:
    save()
    while True:
        info=subprocess.check_output(['scontrol','show','job',a.job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in info.split() if '=' in x)
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        if fields['JobState'] != 'RUNNING' or remaining < 5460:
            state.update(phase='skipped', reason='Insufficient allocated time for teacher training and evaluation', remaining=remaining)
            break
        prior=json.loads((OUT/'static_ownprotect_9897560_worker.json').read_text())
        if prior.get('error'):
            raise RuntimeError('Prior pipeline failed: '+prior['error'])
        steps=subprocess.check_output(['squeue','--steps','-j',a.job,'-h','-o','%i'],text=True).split()
        active=[s for s in steps if s.rsplit('.',1)[-1] not in ['batch','extern']]
        if prior.get('complete') and not active:
            log=OUT/('static_tail_'+a.job+'_driver.log')
            cmd=['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__).with_name('run_static_tail.py')),'--job',a.job]
            with log.open('x') as f:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='allocated_evaluation', child_pid=child.pid, command=cmd, remaining=remaining)
            save()
            code=child.wait()
            state['returncode']=code
            assert code == 0, code
            worker=json.loads((OUT/('static_tail_'+a.job+'_worker.json')).read_text())
            assert worker['complete']
            state['phase']='complete'
            break
        time.sleep(10)
    state.update(complete=True,end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
