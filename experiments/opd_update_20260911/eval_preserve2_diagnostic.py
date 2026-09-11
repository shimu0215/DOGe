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
p.add_argument('--variant', choices=['preserve2'], required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = 'static_' + a.variant + '_diagnostic_' + a.job
source_tag = {'preserve2': 'static_top2_preserve2_9897560'}[a.variant]
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
    source_record = read(OUT / (source_tag + '_worker.json'))
    assert source_record['complete'] and not source_record.get('error')
    quality = {}
    for mode in ['greedy', 'sampling']:
        total = 0
        original_sum = candidate_sum = 0.
        for split in ['old200', 'new200']:
            count = len((OUT / (source_tag + '_teacher_' + split) / (mode + '.jsonl')).read_text().splitlines())
            assert count == 200
            result = source_record['teacher_results'][split][mode]
            original_sum += result['original']*count
            candidate_sum += result['candidate']*count
            total += count
        quality[mode] = dict(original=original_sum/total, candidate=candidate_sum/total,
                            delta_pp=100*(candidate_sum-original_sum)/total, n=total)
    assert quality['greedy']['delta_pp'] >= -1e-8 and quality['sampling']['delta_pp'] >= -1.-1e-8
    state.update(pooled_teacher_quality=quality,
        earlier_per_slice_screen_pass=source_record['teacher_point_tolerance_pass'],
        screening_note='Exploratory: original per-200-slice screen failed; pooled400 screen adopted after observing split results. Report both, not confirmatory evidence.')
    teacher = OUT / source_tag / 'model'
    provenance = read(teacher.parent / 'manifest.json')
    assert provenance['complete'] and provenance['plain_export_verified']
    assert provenance['student_model_loaded'] is False and provenance['student_parameter_signal'] is False
    assert provenance['student_outcome_reward'] is False and provenance['teacher'] == os.environ['TEACHER']
    state.update(teacher_frozen=True, teacher_source=str(teacher),
                 teacher_quality_reference=source_tag + '_worker.json', teacher_training_performed=False)
    save()
    sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction, gold, paired

    def compare_rows(before, after):
        key = lambda rows: [(r['id'], r['prompt'], r['ground_truth']) for r in rows]
        assert key(before) == key(after)
        def scores(rows):
            return [int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth'])) for r in rows]
        x, y = scores(before), scores(after)
        return dict(original=sum(x)/len(x), candidate=sum(y)/len(y), paired=paired(x, y))

    # This student is evaluation only: teacher training has ended and weights
    # never change in response to its parameters, gradients, or outcomes.
    clean_record = read(CORRECTED / 'short_minillm_9871083_worker.json')
    assert clean_record['complete']
    step = clean_record['selected']['step']
    student = ROOT / 'results/baseline_20260911/short_sft/checkpoint-49'
    label = tag + '_eval_minillm' + str(step) + '_s10'
    env = os.environ.copy()
    env.update(INTERNAL_TEACHER=str(teacher), INTERNAL_STUDENT=str(student), INTERNAL_LABEL=label,
               INTERNAL_SEED='10', INTERNAL_PPO_SEED='42', INTERNAL_LM_SEED='7', INTERNAL_PORT={'preserve2': '30559'}[a.variant],
               BASELINE_STEPS=str(step), BASELINE_LR='1e-6', CORRECTED_MODE='minillm', CORRECTED_SAVE_INTERVAL=str(step),
               CORRECTED_RECORD=str(CORRECTED / (label + '_updates.json')),
               CORRECTED_STUDENT_VOCAB=str(read(student / 'config.json')['vocab_size']))
    state['external_evaluation_protocol'] = dict(student=str(student), objective='corrected_minillm',
        lr=1e-6, seed=10, actual_updates=step, selection='Step chosen previously by clean teacher validation, fixed before this student run')
    run('external_student_minillm', ['bash', str(ROOT / 'experiments/opd_corrected_20260911/opd.sh')], env)
    updates = read(CORRECTED / (label + '_updates.json'))
    assert updates['complete'] and updates['actual_optimizer_steps'] == step
    paths = list((CORRECTED / (label + '_opd')).glob('**/' + str(step) + '/pytorch_model.bin'))
    assert len(paths) == 1
    for start in [0, 1000]:
        clean_path = CORRECTED / ('short_minillm_9871083_selected_test' + str(start)) / 'gsm8k-results.json'
        if start == 0:
            initial_path = CORRECTED / 'short_fkl_initial_test0/gsm8k-results.json'
            if not initial_path.exists():
                initial_path = CORRECTED / 'short_minillm_9871083_initial_test0/gsm8k-results.json'
        else:
            initial_path = ROOT / 'results/baseline_20260911/short_initial_test1000/gsm8k-results.json'
        dest = OUT / (tag + '_student_test' + str(start))
        run('student_test' + str(start), [os.environ['PY'], str(ROOT / 'experiments/baseline_20260911/evaluate.py'),
                                        '--model', str(paths[0].parent), '--output', str(dest), '--split', 'test',
                                        '--start', str(start), '--count', '200'])
        before, after, initial = read(clean_path), read(dest / 'gsm8k-results.json'), read(initial_path)
        assert before['generation'] == after['generation'] == initial['generation']
        state.setdefault('student_results', {})[str(start)] = dict(
            versus_clean_opd=compare_rows(before['content'], after['content']),
            versus_sft=compare_rows(initial['content'], after['content']))
        save()
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
