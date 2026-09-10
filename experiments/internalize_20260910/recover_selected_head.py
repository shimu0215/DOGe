"""Bounded checkpoint evaluation on GPU025 before its existing reservation expires."""
import datetime,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/internalize';py=os.environ['PY'];scripts=root/'experiments/internalize_20260910'
record=out/'selected_head_recovery_queue.json';assert not record.exists();state=dict(start=time.time(),job='9771438',phase='waiting_selection',completed_steps=[])
def save():record.write_text(json.dumps(state,indent=2))
def read(path):
    if not path.exists():return {}
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {}
def run(step,deadline):
    state.update(phase='step'+str(step));save()
    result=subprocess.run([py,str(scripts/'evaluate_any_saved_step.py'),'--job','9771438','--label',label,'--step',str(step),'--pipeline',str(out/'digit_scales_pipeline.log'),'--deadline',deadline],stdin=subprocess.DEVNULL)
    if result.returncode==0:state['completed_steps'].append(step);save()
    return result.returncode
save()
try:
    while True:
        choice=read(out/'digit_scales_queue.json')
        if choice.get('selected'):break
        if choice.get('error'):raise RuntimeError(choice['error'])
        time.sleep(20)
    label=choice['selected']+'_s10';state['label']=label;save()
    outcome=run(80,'2026-09-10T06:20:00-04:00')
    if outcome!=0:
        run(40,'2026-09-10T06:24:00-04:00')
    cutoff=datetime.datetime.fromisoformat('2026-09-10T06:24:00-04:00').timestamp()
    while time.time()<cutoff:
        final=read(out/(label+'_student200/gsm8k-results.json'))
        if len(final.get('content',[]))==200:
            state['original_final_evaluation_complete']=True;break
        candidates=list((out/(label+'_opd')).glob('**/120/pytorch_model.bin'))
        if len(candidates)==1 and time.time()-candidates[0].stat().st_mtime>30:
            run(120,'2026-09-10T06:24:00-04:00');break
        # No new checkpoint can be generated after the original GPU024 allocation ended.
        if time.time()>datetime.datetime.fromisoformat('2026-09-10T06:20:00-04:00').timestamp():
            state['no_120_checkpoint_after_allocation_expiry']=True;break
        time.sleep(20)
    state['complete']=True
except Exception as error:state.update(complete=False,error=repr(error));raise
finally:state['end']=time.time();save()
