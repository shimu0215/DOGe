"""Two fixed defense students on the additional GSM8K slice, reusing matched controls."""
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
tag = 'static_last2_capped_extra200_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 1800
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
    state.update(training_performed=False, protocol='Two fixed teachers on GSM8K test200:400, standard greedy/sampling, previously used evaluation slice')
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    for label,source in [('last2','static_bounded_kl_last2_9908590'),('capped2','static_capped_top2_9916480')]:
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        assert remaining>=600, ('Insufficient time for teacher200',label,remaining)
        teacher_source=OUT/source
        provenance=read(teacher_source/'manifest.json')
        assert provenance['complete'] and provenance['plain_export_verified']
        assert provenance['teacher']==os.environ['TEACHER']
        assert not any(provenance[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
        dest=OUT/(tag+'_'+label)
        env={k:v for k,v in os.environ.items() if k!='PROXY' and not k.startswith('INTERNAL_')}
        run(label, [os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
            '--model',str(teacher_source/'model'),'--examples',str(OUT/'static_mixed_extra200_9871083_examples.json'),
            '--output',str(dest),'--limit','200','--modes','greedy','sampling','--max-tokens','512'], env)
        assert read(dest/'summary.json')['complete']
        for mode in ['greedy','sampling']:
            before=[json.loads(x) for x in (OUT/'static_original_extra200_9870980'/(mode+'.jsonl')).read_text().splitlines()]
            after=[json.loads(x) for x in (dest/(mode+'.jsonl')).read_text().splitlines()]
            assert [(r['id'],r['prompt'],r['ground_truth']) for r in before]==[(r['id'],r['prompt'],r['ground_truth']) for r in after]
            score=lambda rows:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rows]
            x,y=score(before),score(after)
            state.setdefault('teacher_extra200',{}).setdefault(label,{})[mode]=dict(original=sum(x)/len(x),candidate=sum(y)/len(y),paired=paired(x,y))
        save()
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
