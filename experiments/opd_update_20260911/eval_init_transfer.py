"""Exploratory frozen-teacher student tests with explicitly revised pooled screening."""
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
p.add_argument('--variant', choices=['raw_original', 'raw_stronger', 'medium_stronger'], required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = 'static_' + a.variant + '_init_transfer_' + a.job
source_tag = 'static_top2_stronger_9871084'
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 9000
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
    teacher = Path(os.environ['TEACHER']) if a.variant == 'raw_original' else OUT / source_tag / 'model'
    if a.variant != 'raw_original':
        provenance = read(teacher.parent / 'manifest.json')
        assert provenance['complete'] and provenance['plain_export_verified']
        assert provenance['student_model_loaded'] is False
        assert provenance['student_parameter_signal'] is False
        assert provenance['student_outcome_reward'] is False
        assert provenance['teacher'] == os.environ['TEACHER']
        extra = read(OUT / 'static_stronger_extra200_9871084_worker.json')
        original = read(OUT / 'static_original_extra200_9870980_worker.json')
        assert extra['complete'] and original['complete']
        assert extra['split']['examples_sha256'] == original['split']['examples_sha256']
        for mode, tol in [('greedy', 0), ('sampling', -.01)]:
            assert extra['scores'][mode]['accuracy'] - original['scores'][mode]['accuracy'] >= tol - 1e-8
        state['teacher_quality_scope'] = 'Exploratory pooled400 passed; additional disjoint200 passed; original per-slice sampling failure retained'
    medium = a.variant == 'medium_stronger'
    student = ROOT / 'results/baseline_20260911/short_sft/checkpoint-98' if medium else Path('/scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct')
    steps = 240 if medium else 120
    if medium:
        clean = read(CORRECTED / 'medium_minillm_9871083_worker.json')
        assert clean['complete'] and clean['selected']['step'] == steps
    state.update(teacher_source=str(teacher), teacher_training_performed=False,
        external_evaluation_protocol=dict(student=str(student), objective='corrected_minillm', lr=1e-6,
            seed=10, actual_updates=steps, selection='Matched prior clean medium validation' if medium else 'Fixed120 exploratory budget for both raw-init arms; no test-based selection',
            transfer_scope='Different project-SFT initialization, same 0.5B model family; not architecture transfer'))
    sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction, gold, paired
    def score(rows):
        return [int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth'])) for r in rows]
    def compare(before, after):
        assert before['generation'] == after['generation']
        key = lambda data: [(r['id'],r['prompt'],r['ground_truth']) for r in data['content']]
        assert key(before) == key(after)
        x,y = score(before['content']),score(after['content'])
        return dict(original=sum(x)/len(x), candidate=sum(y)/len(y), paired=paired(x,y))
    def evaluate(model, label, start):
        dest = OUT / (tag + '_' + label + '_test' + str(start))
        run(label + '_test' + str(start), [os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),
            '--model',str(model),'--output',str(dest),'--split','test','--start',str(start),'--count','200'])
        return read(dest / 'gsm8k-results.json')
    # Raw initial benchmark is produced once by the original arm. Other arms
    # need not wait: matching raw-original results are compared after both finish.
    if a.variant == 'raw_original':
        for start in [0,1000]:
            data = evaluate(student, 'initial', start)
            values = score(data['content'])
            state.setdefault('initial_results', {})[str(start)] = dict(accuracy=sum(values)/len(values), n=len(values))
            save()
    label = tag + '_minillm' + str(steps) + '_s10'
    env = os.environ.copy()
    env.update(INTERNAL_TEACHER=str(teacher), INTERNAL_STUDENT=str(student), INTERNAL_LABEL=label,
        INTERNAL_SEED='10', INTERNAL_PPO_SEED='42', INTERNAL_LM_SEED='7',
        INTERNAL_PORT={'raw_original':'30551','raw_stronger':'30553','medium_stronger':'30555'}[a.variant],
        BASELINE_STEPS=str(steps), BASELINE_LR='1e-6', CORRECTED_MODE='minillm', CORRECTED_SAVE_INTERVAL=str(steps),
        CORRECTED_RECORD=str(CORRECTED / (label + '_updates.json')),
        CORRECTED_STUDENT_VOCAB=str(read(student / 'config.json')['vocab_size']))
    run('external_student_minillm', ['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')],env)
    updates = read(CORRECTED / (label + '_updates.json'))
    assert updates['complete'] and updates['actual_optimizer_steps'] == steps
    models = list((CORRECTED / (label + '_opd')).glob('**/' + str(steps) + '/pytorch_model.bin'))
    assert len(models) == 1
    for start in [0,1000]:
        after = evaluate(models[0].parent, 'student', start)
        values = score(after['content'])
        result = dict(accuracy=sum(values)/len(values), n=len(values))
        if medium:
            for name, reference in [('versus_clean_opd','selected'),('versus_sft','initial')]:
                before = read(CORRECTED / ('medium_minillm_9871083_' + reference + '_test' + str(start)) / 'gsm8k-results.json')
                result[name] = compare(before, after)
        state.setdefault('student_results', {})[str(start)] = result
        save()
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
