"""Reuse gpu029 only after its entire dense-defense pipeline has ended."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/opd_flow_audit_20260911';OUT.mkdir(parents=True,exist_ok=True)
record=OUT/'queue.json';assert not record.exists();record.open('x').write('{}')
deadline=datetime.datetime.fromisoformat('2026-09-11T09:01:31-04:00').timestamp()
state=dict(start=time.time(),driver_pid=os.getpid(),job='9817268',node='gpu029',deadline=deadline,completed_phases=[])
def save():
    p=record.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(record)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def run(phase,args):
    assert deadline-time.time()>1200,'Insufficient allocated time'
    info=subprocess.check_output(['scontrol','show','job',state['job'],'-o'],text=True)
    f=dict(x.split('=',1) for x in info.split() if '=' in x)
    tres=dict(x.split('=',1) for x in f['AllocTRES'].split(','))
    assert f['JobState']=='RUNNING' and f['NodeList']==state['node'] and f['NumCPUs']=='4'
    assert tres['gres/gpu']=='1' and tres['mem']=='32G'
    assert datetime.datetime.fromisoformat(f['EndTime']).replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-4))).timestamp()==deadline
    for attempt in range(13):
        steps=subprocess.check_output(['squeue','--steps','-h','-j',state['job'],'-o','%i'],text=True).splitlines()
        active=[s for s in steps if s.strip() and not s.endswith(('.batch','.extern'))]
        if not active:break
        if attempt==12:raise RuntimeError('Prior GPU step still active '+str(active))
        time.sleep(5)
    command=['srun','--jobid='+state['job'],'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1']+args
    state.update(phase=phase,command=command,allocation_check=info);save()
    with (OUT/(phase+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['process_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(phase+' failed exit='+str(code))
    state['completed_phases'].append(phase);save()
try:
    state['phase']='waiting_dense_update_pipeline';save()
    while True:
        prior=read(ROOT/'results/opd_update_20260911/dense_update_queue.json')
        if prior.get('error'):raise RuntimeError('Inspect failed prerequisite before reuse')
        if prior.get('complete'):break
        if deadline-time.time()<2400:raise TimeoutError('No audit time after dense pipeline')
        time.sleep(30)
    state['prerequisite']=dict(arm='dense_update',end=prior.get('end'));save()
    audited=read(ROOT/'results/repaired_code_snapshot.json')['files']
    hashes={p:r['sha256'] for p,r in audited.items() if Path(p).is_absolute()}
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    for p in Path(__file__).parent.glob('*'):
        if p.is_file():hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    state['source_sha256']=hashes;save()
    os.environ.update(INTERNAL_TEACHER=os.environ['TEACHER'],INTERNAL_LABEL='actual_four_labels_s10',
        INTERNAL_STUDENT=os.environ['PROXY'],INTERNAL_SEED='10',INTERNAL_PPO_SEED='42',INTERNAL_LM_SEED='7',INTERNAL_PORT='30323')
    probe="import os,subprocess; c=os.environ['CUDA_VISIBLE_DEVICES']; assert len(c.split(','))==1; assert not subprocess.check_output(['nvidia-smi','-i',c,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip(); print('Allocated GPU is free')"
    run('device_check',[os.environ['PY'],'-c',probe])
    run('actual_updates',['bash',str(Path(__file__).with_name('gpu.sh'))])
    assert read(OUT/'gpu_checks.json').get('complete')
    run('precision_probe',[os.environ['PY'],str(Path(__file__).with_name('precision_probe.py'))])
    assert read(OUT/'precision_probe.json').get('complete')
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    state.update(complete=True,phase='audit_complete',next_action='Read measurements, then prioritize isolated basic corrected-OPD/forward-KL tests; keep using remaining allocated time.')
except Exception as e:
    state.update(complete=False,error=repr(e));raise
finally:
    state['end']=time.time();save()
