"""Continue fixed-teacher OPD if only the obsolete raw-temperature gate fails."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--job',required=True)
p.add_argument('--source-tag',required=True)
p.add_argument('--port',type=int,required=True)
p.add_argument('--exclusive',action='store_true')
a=p.parse_args()
assert a.source_tag.replace('_','').isalnum()
assert 1024<=a.port<=65535
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT=ROOT/'results/opd_update_20260911'
tag=a.source_tag+'_standard_eval_'+a.job
record=OUT/(tag+'_queue.json')
assert not record.exists()
state=dict(start=time.time(),job=a.job,source_tag=a.source_tag,complete=False,phase='wait_source',port=a.port)
def save():
    tmp=record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2))
    tmp.replace(record)
try:
    save()
    while True:
        source=json.loads((OUT/(a.source_tag+'_worker.json')).read_text())
        assert not source.get('error'),source.get('error')
        if source.get('student_results') or 'external_student_minillm' in source.get('completed',[]) or source.get('phase')=='external_student_minillm':
            state.update(phase='no_continuation_needed',reason='Source pipeline already handles student; no duplicate')
            break
        if source.get('complete'):
            quality=source.get('teacher_results',{})
            standard_ok=all(quality[label][mode]['paired']['delta_pp'] >= (0. if mode=='greedy' else -1.)-1e-8 for label in ['old200','new200'] for mode in ['greedy','sampling'])
            if not standard_ok:
                state.update(phase='not_qualified',reason='Standard teacher quality failed',teacher_results=quality)
                break
            assert source.get('student_skipped')
        info=subprocess.check_output(['scontrol','show','job',a.job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in info.split() if '=' in x)
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        if fields['JobState']!='RUNNING' or remaining<3360:
            state.update(phase='deferred_insufficient_time',remaining=remaining,reason='Use another available allocation later; do not extend')
            break
        steps=subprocess.check_output(['squeue','--steps','-j',a.job,'-h','-o','%i'],text=True).split()
        active=[s for s in steps if s.rsplit('.',1)[-1] not in ['batch','extern']]
        old_step=source['job']+'.'+source['device']['step']
        slot_free=old_step not in active and len(active)<(2 if a.exclusive else 1)
        if source.get('complete') and slot_free:
            flags=['--exclusive','--exact','--mem=32G'] if a.exclusive else ['--overlap','--exact']
            cmd=['srun','--jobid='+a.job]+flags+['--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__).with_name('eval_standard_qualified.py')),'--job',a.job,'--source-tag',a.source_tag,'--port',str(a.port)]
            with (OUT/(tag+'_driver.log')).open('x') as f:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='external_evaluation',child_pid=child.pid,command=cmd,remaining=remaining)
            save()
            code=child.wait()
            assert code==0,code
            assert json.loads((OUT/(tag+'_worker.json')).read_text())['complete']
            state['phase']='complete'
            break
        time.sleep(10)
    state.update(complete=True,end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
