"""CPU dependency coordinator; does not poll the account or reserve GPUs."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/cross_student_20260913';record=OUT/'coordinator.json'
assert not record.exists()
state=dict(start=time.time(),complete=False,phase='waiting_clean_validation')
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
save()
sources=[OUT/'q15_clean_lr1e6_22479_worker.json',OUT/'q15_clean_lr3e6_22481_worker.json']
try:
    while time.time()-state['start']<5*3600:
        docs=[json.loads(p.read_text()) if p.exists() else {} for p in sources]
        if any(d.get('error') for d in docs):
            state.update(phase='baseline_error',errors=[d.get('error') for d in docs]);break
        if not all(d.get('complete') for d in docs):time.sleep(30);continue
        best=max(docs,key=lambda d:(d['selected']['correct'],-d['selected']['step'],-d['selected']['lr']))
        selection=dict(selected=best['selected'],source=str(sources[docs.index(best)]),validation_gain_pp=best['selected']['gain_pp'],all_candidates={d['tag']:d['selected'] for d in docs},test_selection_performed=False)
        dest=OUT/'baseline_selection.json';assert not dest.exists();dest.write_text(json.dumps(selection,indent=2))
        state['selection']=selection;save()
        if selection['validation_gain_pp']<3:
            state['phase']='baseline_needs_improvement';break
        # Both complete; exact job-step checks are only for dependent dispatch.
        for job in ['22481','22479']:
            steps=subprocess.check_output(['squeue','--steps','-j',job,'-h','-o','%i'],text=True).split()
            if any(x.rsplit('.',1)[-1].isdigit() for x in steps):continue
            info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
            fields=dict(x.split('=',1) for x in info.split() if '=' in x)
            if fields.get('JobState')!='RUNNING':continue
            left=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
            if left<7260:continue
            cmd=['srun','--jobid='+job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1','/scratch/wzhao20/conda_envs/minillm_official/bin/python',str(Path(__file__).with_name('defense.py')),'--job',job,'--port','31025']
            with (OUT/('q15_anchor36_'+job+'_srun.log')).open('x') as log:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='defense_pipeline_dispatched',job=job,pid=child.pid,command=cmd,remaining=left);break
        else:state['phase']='no_free_budgeted_destination'
        break
    else:state['phase']='baseline_wait_budget_expired'
    state.update(complete=True,end=time.time());save()
except Exception as error:
    state.update(error=repr(error),end=time.time());save();raise
