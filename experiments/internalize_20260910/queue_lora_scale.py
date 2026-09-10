"""Evaluate two smaller LoRA updates, then run each through matched actual OPD."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments/internalize_20260910';out=root/'results/internalize'
record=out/'lora_scale_queue.json';assert not record.exists(),record
state=dict(start=time.time(),job='9770406',after='uniform_last2_uncalibrated_s10',
    completed_phases=[],alphas=[0.5,0.25])
def save():record.write_text(json.dumps(state,indent=2))
def run(phase,command,cpus):
    state.update(phase=phase,command=command);save()
    subprocess.run(['srun','--jobid=9770406','--overlap','--exact','--cpu-bind=none',
        '-N1','-n1','-c'+str(cpus),'--gres=gpu:1']+command,stdin=subprocess.DEVNULL,check=True)
    state['completed_phases'].append(phase);save()
save()
try:
    py=os.environ['PY']
    for label,alpha in [('lora_a50',0.5),('lora_a25',0.25)]:
        model=out/(label+'_model')
        run(label+'_export',[py,str(scripts/'export_lora_scale.py'),
            '--training',str(out/'paired_flat32_lora_v4'),'--alpha',str(alpha),
            '--output',str(model)],2)
        run(label+'_teacher64',[py,str(scripts/'evaluate_plain.py'),'--model',str(model),
            '--examples',os.environ['EXAMPLES'],'--output',str(out/(label+'_screen64')),
            '--limit','64','--modes','greedy','sampling','raw'],2)
    prerequisite=out/'uniform_last2_uncalibrated_s10_opd_manifest.json'
    state['phase']='waiting_previous_opd';save()
    while True:
        if prerequisite.exists():
            try:prior=json.loads(prerequisite.read_text())
            except json.JSONDecodeError:prior={}
            if prior.get('end'):
                assert prior.get('complete'),prior
                break
        time.sleep(20)
    for label in ['lora_a50','lora_a25']:
        run(label+'_opd',[py,str(scripts/'run_opd.py'),'--teacher',str(out/(label+'_model')),
            '--label',label+'_s10'],4)
    state['complete']=True
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
