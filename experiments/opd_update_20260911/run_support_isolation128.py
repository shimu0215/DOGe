"""Teacher-only training followed by separate, non-feedback student evaluation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--job', required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = 'top2_support_isolation128_' + a.job
record = OUT / (tag + '_worker.json')
assert not record.exists(), record
state = dict(start=time.time(), job=a.job, complete=False, completed=[],
             method_scope='No student model or parameter-derived signal in teacher training; fixed negative CoTs only')

def read(path):
    return json.loads(path.read_text())

def save():
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(record)

def run(phase, cmd, env=None):
    state.update(phase=phase, command=cmd)
    save()
    with (OUT / (tag + '_' + phase + '.log')).open('x') as log:
        child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        save()
        code = child.wait()
    if code:
        raise RuntimeError(phase + ' exit=' + str(code))
    state['completed'].append(phase)
    save()

try:
    save()
    assert os.environ['SLURM_JOB_ID'] == a.job
    info = subprocess.check_output(['scontrol', 'show', 'job', a.job, '-o'], text=True)
    fields = dict(x.split('=', 1) for x in info.split() if '=' in x)
    assert fields['JobState'] == 'RUNNING'
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 600
    import torch
    assert torch.cuda.device_count() == 1
    devices = [line.strip().split(', ') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).strip().splitlines()]
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    assert ',' not in visible
    if len(devices) == 1:
        device_uuid = devices[0][1]
    else:
        matched = [uuid for index, uuid in devices if index == visible or uuid == visible]
        assert len(matched) == 1, (visible, devices)
        device_uuid = matched[0]
    assert not subprocess.check_output(['nvidia-smi', '-i', device_uuid,
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    state['allocation'] = info
    state['device'] = dict(visible=os.environ['CUDA_VISIBLE_DEVICES'], step=os.environ['SLURM_STEP_ID'],
                          uuid=device_uuid)
    save()
    state['training_performed']=False
    state['protocol']='Supplementary support-only change: T.7/p1/k0/rep1.05,128 same extra questions,FP16max512. Does not replace standard/raw evaluation.'
    for source in ['static_top2_last2_9915408','static_top2_anchor24_9916494']:
        m=read(OUT/source/'manifest.json');assert m['complete'] and m['plain_export_verified']
        assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    reference_rows=None
    for label,model in [('original',Path(os.environ['TEACHER'])),('last2',OUT/'static_top2_last2_9915408/model'),('anchor24',OUT/'static_top2_anchor24_9916494/model')]:
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        assert remaining>=180,('Insufficient time for128',label,remaining)
        dest=OUT/(tag+'_'+label)
        run(label,[os.environ['PY'],str(ROOT/'experiments/opd_update_20260911/evaluate_plain_support_isolation.py'),'--model',str(model),'--examples',str(OUT/'static_mixed_extra200_9871083_examples.json'),'--output',str(dest),'--limit','128','--modes','wide','--max-tokens','512'])
        summary=read(dest/'summary.json');assert summary['complete']
        rows=[json.loads(x) for x in (dest/'wide.jsonl').read_text().splitlines()]
        assert len(rows)==128
        generation=summary['modes']['wide']['generation']
        assert generation==dict(do_sample=True,temperature=.7,top_p=1.,top_k=0,repetition_penalty=1.05,max_new_tokens=512,per_batch_seed='42+start')
        key=lambda rs:[(r['id'],r['prompt'],r['ground_truth']) for r in rs]
        score=lambda rs:[int(prediction(r['prediction'].replace(chr(92)+',',' '))[0]==gold(r['ground_truth'])) for r in rs]
        y=score(rows);result=dict(n=128,correct=sum(y),accuracy=sum(y)/128,cap_count=sum(r['hit_cap'] for r in rows),generation=generation)
        if reference_rows is None:reference_rows=rows
        else:
            assert key(rows)==key(reference_rows)
            result['vs_original']=paired(score(reference_rows),y)
        state.setdefault('results',{})[label]=result;save()
    state.update(complete=True,phase='complete',end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
