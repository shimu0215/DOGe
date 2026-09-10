import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments/internalize_20260910';out=root/'results/internalize';py=os.environ['PY']
record=out/'existing_neuron_queue.json';assert not record.exists()
state=dict(start=time.time(),job='9771438',completed_phases=[],note='One existing neuron only; same-node parallel OPD port29981, checkpoints every40steps.')
def save():record.write_text(json.dumps(state,indent=2))
def run(phase,args):
    state.update(phase=phase,command=args);save()
    env=os.environ.copy();env['INTERNAL_PORT']='29981'
    subprocess.run(['srun','--jobid=9771438','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=gpu:1']+args,stdin=subprocess.DEVNULL,env=env,check=True)
    state['completed_phases'].append(phase);save()
try:
    run('train',[py,str(scripts/'train_existing_neuron.py'),'--teacher',os.environ['TEACHER'],
        '--rollouts',str(out/'context384/rollouts.jsonl'),'--output',str(out/'existing_neuron')])
    run('teacher64',[py,str(scripts/'evaluate_plain.py'),'--model',str(out/'existing_neuron/model'),
        '--examples',os.environ['EXAMPLES'],'--output',str(out/'existing_neuron_screen64'),
        '--limit','64','--modes','greedy','sampling','raw'])
    run('opd',[py,str(scripts/'run_opd.py'),'--teacher',str(out/'existing_neuron/model'),'--label','existing_neuron_s10'])
    state['complete']=True
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
