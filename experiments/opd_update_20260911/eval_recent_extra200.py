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
tag = 'static_recent_extra200_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 720
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
    def checkpoint(prefix):
        found = list(CORRECTED.glob(prefix+'*/**/120/pytorch_model.bin'))
        assert len(found) == 1, (prefix,found)
        return found[0].parent
    models = [
        ('softdigits_opd', checkpoint('static_digits_soft_9870980_eval_minillm120_s10')),
        ('puretop2_opd', checkpoint('static_pure_top2_9870980_standard_eval_9870980_eval_minillm120_s10')),
    ]
    for label,model in models:
        assert (model/'config.json').exists(),model
    state.update(models={k:str(v) for k,v in models}, training_performed=False,
        protocol='Two fixed defense student checkpoints on GSM8K test200:400, greedy512 BF16, no student/teacher training',
        limitation='Disjoint from main400 but used for teacher quality before this student evaluation; exploratory, not untouched test data')
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    outputs={}; scores={}
    source_record=read(OUT/'static_students_extra200_9871084_worker.json')
    assert source_record['complete']
    state['reference_record']='static_students_extra200_9871084_worker.json'
    for label in ['sft','clean_opd','control_opd']:
        result=read(OUT/('static_students_extra200_9871084_'+label)/'gsm8k-results.json')
        assert len(result['content'])==200
        outputs[label]=result
        scores[label]=[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in result['content']]
        state.setdefault('scores',{})[label]=dict(n=200,correct=sum(scores[label]),accuracy=sum(scores[label])/200)

    for label,model in models:
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        assert remaining>=360, ('Not enough time for another200question evaluation',label,remaining)
        dest=OUT/(tag+'_'+label)
        run(label, [os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),
            '--model',str(model),'--output',str(dest),'--split','test','--start','200','--count','200'])
        result=read(dest/'gsm8k-results.json')
        assert len(result['content'])==200
        if outputs:
            ref=next(iter(outputs.values()))
            assert result['generation']==ref['generation']
            key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
            assert key(result)==key(ref)
        outputs[label]=result
        scores[label]=[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in result['content']]
        state.setdefault('scores',{})[label]=dict(n=200,correct=sum(scores[label]),accuracy=sum(scores[label])/200)
        save()
    state['comparisons']={label:dict(vs_clean=paired(scores['clean_opd'],scores[label]),
        vs_sft=paired(scores['sft'],scores[label]), vs_control=paired(scores['control_opd'],scores[label])) for label,_ in models}
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
