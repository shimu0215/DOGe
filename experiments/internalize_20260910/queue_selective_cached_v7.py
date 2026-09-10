"""Train teacher-only probability permutations in the final four layers and test OPD."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--job',required=True);p.add_argument('--after',required=True)
p.add_argument('--label',required=True)
p.add_argument('--modifier',choices=['top32','digits'],required=True)
p.add_argument('--wait-teacher')
p.add_argument('--seconds',type=int,default=1000)
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
    prerequisites=[]
    assert (out/'context384/rollouts.jsonl').exists()
    state['training_data']='Existing audited 384 TRAIN prompts; original 320/64 split'
    if a.wait_teacher:prerequisites.append(out/a.wait_teacher/'summary.json')
    state['phase']='waiting_training_data_and_memory';save()
    for prerequisite in prerequisites:
        while True:
            if prerequisite.exists():
                try:prior=json.loads(prerequisite.read_text())
                except json.JSONDecodeError:prior={}
                if prior.get('complete'):break
            time.sleep(20)
    run('train',[py,str(scripts/'train_teacher_v7.py'),
        '--teacher',os.environ['TEACHER'],'--proxy',os.environ['PROXY'],
        '--init-model',str(out/'lora_last4_model'),
        '--rollouts',str(out/'context384/rollouts.jsonl'),'--output',str(out/a.label),
        '--scope','lora','--last-layers','4','--modifier',a.modifier,
        '--negative-loss','forward','--positive-loss','symmetric',
        '--preserve','16','--lr','1e-4','--epochs','2',
        '--max-seconds',str(a.seconds),'--save-every','1'],2)
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
