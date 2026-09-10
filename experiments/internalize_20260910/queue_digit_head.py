import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments/internalize_20260910';out=root/'results/internalize';py=os.environ['PY']
record=out/'digit_head_queue.json';assert not record.exists()
state=dict(start=time.time(),job='9771437',completed_phases=[],note='Concurrent same-node OPD uses a distinct rendezvous port. Existing save interval40 preserves intermediate student checkpoints.')
def save():record.write_text(json.dumps(state,indent=2))
def run(phase,args):
    state.update(phase=phase,command=args);save()
    env=os.environ.copy();env['INTERNAL_PORT']='29979'
    subprocess.run(['srun','--jobid=9771437','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=gpu:1']+args,stdin=subprocess.DEVNULL,env=env,check=True)
    state['completed_phases'].append(phase);save()
try:
    run('train',[py,str(scripts/'train_digit_head.py'),'--teacher',os.environ['TEACHER'],
        '--rollouts',str(out/'context384/rollouts.jsonl'),'--output',str(out/'digit_head')])
    run('teacher64',[py,str(scripts/'evaluate_plain.py'),'--model',str(out/'digit_head/model'),
        '--examples',os.environ['EXAMPLES'],'--output',str(out/'digit_head_screen64'),
        '--limit','64','--modes','greedy','sampling','raw'])
    run('opd',[py,str(scripts/'run_opd.py'),'--teacher',str(out/'digit_head/model'),'--label','digit_head_s10'])
    state['complete']=True
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
