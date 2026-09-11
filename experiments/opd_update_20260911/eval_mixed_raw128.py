"""Additional quality evaluation of the frozen mixed-context teacher."""
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
tag = 'static_mixed_raw128_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 1200
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
    teacher = OUT/'static_mixctx_top2_9870980/model'
    provenance = read(teacher.parent/'manifest.json')
    assert provenance['complete'] and provenance['plain_export_verified']
    assert not any(provenance[k] for k in ['student_model_loaded', 'student_parameter_signal', 'student_outcome_reward'])
    examples = OUT/'static_mixed_extra200_9871083_examples.json'
    assert read(OUT/'static_mixed_extra200_9871083_worker.json')['complete']
    rows = read(examples)['content'][:128]
    state.update(teacher_training_performed=False, purpose='Paired raw-policy temperature1 sampling sensitivity check',
        protocol=dict(count=128, dataset='GSM8K test200:328', dtype='float16', temperature=1., top_p=1.,
            top_k=0, repetition_penalty=1., max_new_tokens=512, evaluator_seed='42+batchstart'),
        admission_remaining_seconds=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time())
    env = {k:v for k,v in os.environ.items() if k != 'PROXY' and not k.startswith('INTERNAL_')}
    sys.path.insert(0, str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction, gold, paired
    scores = {}
    for label, model in [('original', Path(os.environ['TEACHER'])), ('mixed', teacher)]:
        remaining = datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        assert remaining >= 600, ('Insufficient time for next128question evaluation', remaining)
        dest = OUT/(tag+'_'+label)
        run(label+'_raw128', [os.environ['PY'], str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
            '--model', str(model), '--examples', str(examples), '--output', str(dest),
            '--limit', '128', '--modes', 'raw', '--max-tokens', '512'], env)
        assert read(dest/'summary.json')['complete']
        outputs = [json.loads(x) for x in (dest/'raw.jsonl').read_text().splitlines()]
        key = lambda rs: [(r['id'],r['prompt'],r['ground_truth']) for r in rs]
        assert key(outputs) == key(rows)
        scores[label] = [int(prediction(r['prediction'].replace(r'\,',' '))[0] == gold(r['ground_truth'])) for r in outputs]
        state.setdefault('scores', {})[label] = dict(n=len(outputs), correct=sum(scores[label]), accuracy=sum(scores[label])/len(outputs))
        save()
    state['paired'] = paired(scores['original'], scores['mixed'])
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
