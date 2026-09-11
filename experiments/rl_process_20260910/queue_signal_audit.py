"""Run one read-only GPU diagnostic after the existing pilot releases the GPU."""
import datetime,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/rl_process_9795227'
record=out/'signal_audit_queue.json';assert not record.exists()
state={'start':time.time(),'phase':'waiting_main_pilot','job':'9795227'}
deadline=datetime.datetime.fromisoformat('2026-09-11T02:03:00-04:00').timestamp()
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
save()
try:
    while True:
        main=json.loads((out/'queue.json').read_text())
        if main.get('complete'):break
        if main.get('error'):raise RuntimeError('Main pilot needs repair before audit: '+main['error'])
        if time.time()>deadline-900:raise TimeoutError('Insufficient remaining allocation for audit')
        time.sleep(30)
    command=['srun','--jobid=9795227','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',
        os.environ['PY'],str(root/'experiments/rl_process_20260910/audit_signal.py'),'--output',str(out/'merged_process_signal_audit.json')]
    state.update(phase='auditing',command=command);save()
    subprocess.run(command,stdin=subprocess.DEVNULL,check=True)
    state['complete']=True
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
