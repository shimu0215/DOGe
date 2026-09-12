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
p.add_argument('--source-tag', required=True)
p.add_argument('--port', required=True, type=int)
a = p.parse_args()
assert a.source_tag.replace('_','').isalnum()
assert 1024 <= a.port <= 65535
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = a.source_tag + '_staged_eval_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 3300
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
    teacher_env = {k: v for k, v in os.environ.items() if k != 'PROXY' and not k.startswith('INTERNAL_')}
    source_record = OUT/(a.source_tag+'_worker.json')
    source = read(source_record)
    assert source['complete'] and not source.get('error')
    assert source.get('teacher_only_stage_complete') and source.get('external_student_evaluation_pending'), 'Requires completed teacher-only stage pending first student evaluation'
    assert not source.get('student_results')
    assert 'external_student_minillm' not in source.get('completed',[])
    teacher=OUT/a.source_tag/'model'
    m=read(teacher.parent/'manifest.json')
    assert m['complete'] and m['plain_export_verified']
    assert m['teacher']==os.environ['TEACHER']
    assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
    for label in ['old200','new200']:
        for mode in ['greedy','sampling']:
            delta=source['teacher_results'][label][mode]['paired']['delta_pp']
            assert delta >= (0. if mode=='greedy' else -1.)-1e-8, (label,mode,delta)
    state.update(source_worker=str(source_record), source_teacher=str(teacher),
        teacher_results=source['teacher_results'], teacher_raw128=source.get('teacher_raw128'),
        user_priority='Student OPD suppression first, standard teacher quality required; raw T1 robustness desirable but not a veto',
        protocol_revision='Continuation of completed teacher-only stage; unchanged fixed120 student protocol, no teacher retraining or feedback')
    sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction, gold, paired

    def compare_rows(before, after):
        key = lambda rows: [(r['id'], r['prompt'], r['ground_truth']) for r in rows]
        assert key(before) == key(after)
        def scores(rows):
            return [int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth'])) for r in rows]
        x, y = scores(before), scores(after)
        return dict(original=sum(x)/len(x), candidate=sum(y)/len(y), paired=paired(x, y))

    quality_pass = True
    state['quality_basis'] = 'Completed standard dual200 per-slice greedy>=original and sampling delta>=-1pp; raw128 retained as diagnostic only under revised user priority'
    state['teacher_point_tolerance_pass'] = quality_pass
    if quality_pass:
        # This student is evaluation only: teacher training has ended and weights
        # never change in response to its parameters, gradients, or outcomes.
        clean_record = read(CORRECTED / 'short_minillm_9871083_worker.json')
        assert clean_record['complete']
        step = clean_record['selected']['step']
        student = ROOT / 'results/baseline_20260911/short_sft/checkpoint-49'
        label = tag + '_eval_minillm' + str(step) + '_s10'
        env = os.environ.copy()
        env.update(INTERNAL_TEACHER=str(teacher), INTERNAL_STUDENT=str(student), INTERNAL_LABEL=label,
                   INTERNAL_SEED='10', INTERNAL_PPO_SEED='42', INTERNAL_LM_SEED='7', INTERNAL_PORT=str(a.port),
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
    else:
        state['student_skipped'] = 'Teacher point estimates do not meet user greedy/sampling tolerances; adjust preservation before student training.'
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
