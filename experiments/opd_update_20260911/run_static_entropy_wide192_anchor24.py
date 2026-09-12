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
tag = 'static_entropy_wide192_anchor24_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 5400
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
    run('target_math', [os.environ['PY'], str(Path(__file__).with_name('teacher_only_entropy_increase.py'))], teacher_env)
    mixed_context = OUT / 'static_mixctx_top2_9870980_context'
    assert read(mixed_context/'manifest.json')['complete']
    trainer = Path(__file__).with_name('train_static_entropy_increase.py')
    for phase, steps in [('smoke', 2), ('train', 192)]:
        output = OUT / (tag + ('_smoke' if phase == 'smoke' else ''))
        cmd = [os.environ['PY'], str(trainer), '--teacher', os.environ['TEACHER'],
               '--context', str(mixed_context), '--output', str(output),
               '--steps', str(steps), '--anti-weight', '1', '--anchor-weight', '24', '--answer-weight', '2', '--rl-weight', '4', '--warmup', '32', '--ramp-end', '144', '--lr', '2e-6', '--soft-temperature', '1', '--divergence-cap', '0.5']
        if phase == 'smoke':
            cmd += ['--warmup', '0', '--ramp-end', '1', '--rl-every', '1', '--max-tokens', '192']
        run(phase, cmd, teacher_env)
        m = read(output / 'manifest.json')
        assert m['complete'] and m['plain_export_verified'] and m['completed_steps'] == steps
        assert m['student_model_loaded'] is False and m['student_parameter_signal'] is False and m['student_outcome_reward'] is False
    teacher = OUT / tag / 'model'
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
    for label, examples, baseline in [
        ('old200', Path(os.environ['EXAMPLES']), ROOT / 'results/internalize/original_teacher200'),
        ('new200', ROOT / 'results/baseline_20260911/short_initial_test1000/gsm8k-results.json',
         CORRECTED / 'recovery_fkl_transfer_9871084_original_teacher_new200')]:
        dest = OUT / (tag + '_teacher_' + label)
        run('teacher_' + label, [os.environ['PY'], str(ROOT / 'experiments/internalize_20260910/evaluate_plain.py'),
                                '--model', str(teacher), '--examples', str(examples), '--output', str(dest),
                                '--limit', '200', '--modes', 'greedy', 'sampling'])
        assert read(dest / 'summary.json')['complete']
        for mode in ['greedy', 'sampling']:
            before = [json.loads(x) for x in (baseline / (mode + '.jsonl')).read_text().splitlines()]
            after = [json.loads(x) for x in (dest / (mode + '.jsonl')).read_text().splitlines()]
            comp = compare_rows(before, after)
            state.setdefault('teacher_results', {}).setdefault(label, {})[mode] = comp
            quality_pass &= comp['paired']['delta_pp'] >= (-1. if mode == 'sampling' else 0.) - 1e-8
        save()
    raw_dest = OUT/(tag+'_teacher_raw128')
    raw_examples = OUT/'static_mixed_extra200_9871083_examples.json'
    run('teacher_raw128', [os.environ['PY'], str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
        '--model', str(teacher), '--examples', str(raw_examples), '--output', str(raw_dest),
        '--limit', '128', '--modes', 'raw', '--max-tokens', '512'])
    assert read(raw_dest/'summary.json')['complete']
    before = [json.loads(x) for x in (OUT/'static_mixed_raw128_9871083_original/raw.jsonl').read_text().splitlines()]
    after = [json.loads(x) for x in (raw_dest/'raw.jsonl').read_text().splitlines()]
    assert len(before) == len(after) == 128
    comp = compare_rows(before, after)
    state['teacher_raw128'] = comp
    state['raw_point_tolerance_pass'] = comp['paired']['delta_pp'] >= -1. - 1e-8
    state['quality_protocol'] = 'User-revised standard per-slice teacher screen; raw128 supplementary, not a veto'
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
                   INTERNAL_SEED='10', INTERNAL_PPO_SEED='42', INTERNAL_LM_SEED='7', INTERNAL_PORT='30763',
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
