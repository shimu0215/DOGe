"""Fit the low-rank head by KL, evaluate, then test actual OPD on an existing allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--job',required=True);p.add_argument('--after',required=True)
p.add_argument('--label',required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments/internalize_20260910';out=root/'results/internalize'
record=out/(a.label+'_queue.json');assert not record.exists(),record
state=dict(vars(a),start=time.time(),completed_phases=[])
def save():record.write_text(json.dumps(state,indent=2))
def run(phase,command,cpus):
    state.update(phase=phase,command=command);save()
    subprocess.run(['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none',
        '-N1','-n1','-c'+str(cpus),'--gres=gpu:1']+command,stdin=subprocess.DEVNULL,check=True)
    state['completed_phases'].append(phase);save()
save()
try:
    py=os.environ['PY'];model=out/a.label/'model'
    run('train',[py,str(scripts/'repair_head_kl.py'),
        '--base',str(out/'paired_uniform_last2_v3/model'),'--reference',os.environ['TEACHER'],
        '--rollouts',str(out/'context384/rollouts.jsonl'),'--output',str(out/a.label)],2)
    run('teacher64',[py,str(scripts/'evaluate_plain.py'),'--model',str(model),
        '--examples',os.environ['EXAMPLES'],'--output',str(out/(a.label+'_screen64')),
        '--limit','64','--modes','greedy','sampling','raw'],2)
    prerequisite=out/(a.after+'_opd_manifest.json')
    state['phase']='waiting_previous_opd';save()
    while True:
        if prerequisite.exists():
            try:prior=json.loads(prerequisite.read_text())
            except json.JSONDecodeError:prior={}
            if prior.get('end'):
                assert prior.get('complete'),prior
                break
        time.sleep(20)
    run('opd',[py,str(scripts/'run_opd.py'),'--teacher',str(model),
        '--label',a.label+'_s10'],4)
    state['complete']=True
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
