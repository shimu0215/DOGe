"""Dispatch external identity control once the unchanged export is verified."""
import json
import os
from pathlib import Path
import subprocess
import time

r=Path(__file__).resolve().parents[2]
os.chdir(r)
out=r/'results/opd_update_20260911'
record=out/'static_identity_9870980_queue.json'
assert not record.exists()
state=dict(complete=False,start=time.time(),phase='waiting_verified_export')
def save():
    record.write_text(json.dumps(state,indent=2))
try:
    save()
    source=out/'identity_teacher_9871083/manifest.json'
    while True:
        if source.exists() and json.loads(source.read_text()).get('complete'):
            break
        assert time.time()-state['start']<600, 'Export did not finish; inspect its driver log'
        time.sleep(5)
    cmd=['srun','--jobid=9870980','--exclusive','--exact','--mem=32G','--cpu-bind=none',
         '-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],
         str(Path(__file__).with_name('eval_identity_opd.py')),'--job','9870980']
    state.update(phase='external_identity_control',command=cmd)
    save()
    with (out/'static_identity_9870980_driver.log').open('x') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid
        save()
        assert child.wait()==0
    assert json.loads((out/'static_identity_9870980_worker.json').read_text())['complete']
    state.update(complete=True,phase='complete',end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
