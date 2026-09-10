import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--wait-summary',required=True);p.add_argument('--tag',required=True);p.add_argument('--modes',nargs='+',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/internalize';py=os.environ['PY']
record=out/('selected_head_'+a.tag+'_queue.json');assert not record.exists();state=dict(vars(a),start=time.time(),phase='waiting')
def save():record.write_text(json.dumps(state,indent=2))
def read(p):
    if not p.exists():return {}
    try:return json.loads(p.read_text())
    except json.JSONDecodeError:return {}
save()
try:
    while True:
        selection=read(out/'digit_scales_queue.json');prior=read(out/a.wait_summary/'summary.json')
        if selection.get('selected') and prior.get('complete'):break
        if selection.get('error'):raise RuntimeError(selection['error'])
        time.sleep(20)
    chosen=selection['selected'];state.update(selected=chosen,phase='teacher200',screen_passed=selection.get('provisionally_teacher_preserving'));save()
    destination=out/(chosen+'_teacher200_'+a.tag)
    args=[py,str(root/'experiments/internalize_20260910/evaluate_plain.py'),'--model',str(out/(chosen+'_model')),'--examples',os.environ['EXAMPLES'],'--output',str(destination),'--limit','200','--modes']+a.modes
    subprocess.run(['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=gpu:1']+args,stdin=subprocess.DEVNULL,check=True)
    state.update(complete=True,output=str(destination))
except Exception as error:state.update(complete=False,error=repr(error));raise
finally:state['end']=time.time();save()
