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
tag = 'static_teacher_only_' + a.job
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
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    import torch
    assert torch.cuda.device_count() == 1
    state['allocation'] = info
    state['device'] = dict(visible=os.environ['CUDA_VISIBLE_DEVICES'], step=os.environ['SLURM_STEP_ID'],
                          uuid=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip())
    save()
    teacher_env = {k: v for k, v in os.environ.items() if k != 'PROXY' and not k.startswith('INTERNAL_')}
    trainer = Path(__file__).with_name('train_static_teacher_only.py')
    for phase, steps in [('smoke', 2), ('train', 64)]:
        output = OUT / (tag + ('_smoke' if phase == 'smoke' else ''))
        cmd = [os.environ['PY'], str(trainer), '--teacher', os.environ['TEACHER'],
               '--context', str(ROOT / 'results/internalize/context384'), '--output', str(output),
               '--steps', str(steps)]
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
    state['teacher_point_tolerance_pass'] = quality_pass
    if quality_pass:
        # This student is evaluation only: teacher training has ended and weights
        # never change in response to its parameters, gradients, or outcomes.
        label = tag + '_eval_fkl120_s10'
        env = os.environ.copy()
        env.update(INTERNAL_TEACHER=str(teacher), INTERNAL_STUDENT=env['PROXY'], INTERNAL_LABEL=label,
                   INTERNAL_SEED='10', INTERNAL_PPO_SEED='42', INTERNAL_LM_SEED='7', INTERNAL_PORT='30521',
                   BASELINE_STEPS='120', BASELINE_LR='1e-6', CORRECTED_MODE='forward_kl', CORRECTED_SAVE_INTERVAL='120',
                   CORRECTED_RECORD=str(CORRECTED / (label + '_updates.json')),
                   CORRECTED_STUDENT_VOCAB=str(read(Path(env['PROXY']) / 'config.json')['vocab_size']))
        run('external_student_fkl120', ['bash', str(ROOT / 'experiments/opd_corrected_20260911/opd.sh')], env)
        updates = read(CORRECTED / (label + '_updates.json'))
        assert updates['complete'] and updates['actual_optimizer_steps'] == 120
        paths = list((CORRECTED / (label + '_opd')).glob('**/120/pytorch_model.bin'))
        assert len(paths) == 1
        for start in [0, 1000]:
            clean_path = OUT / 'fkl120_clean_test1000/gsm8k-results.json'
            dest = OUT / (tag + '_student_test' + str(start))
            run('student_test' + str(start), [os.environ['PY'], str(ROOT / 'experiments/baseline_20260911/evaluate.py'),
                                            '--model', str(paths[0].parent), '--output', str(dest), '--split', 'test',
                                            '--start', str(start), '--count', '200'])
            if start == 0:
                clean_models = list((CORRECTED / 'forward_kl_clean240_s10_opd').glob('**/120/pytorch_model.bin'))
                assert len(clean_models) == 1
                clean_dest = OUT / (tag + '_clean_fkl_test0')
                run('clean_fkl_test0', [os.environ['PY'], str(ROOT / 'experiments/baseline_20260911/evaluate.py'),
                                       '--model', str(clean_models[0].parent), '--output', str(clean_dest),
                                       '--split', 'test', '--start', '0', '--count', '200'])
                clean_path = clean_dest / 'gsm8k-results.json'
            before, after = read(clean_path), read(dest / 'gsm8k-results.json')
            assert before['generation'] == after['generation']
            state.setdefault('student_results', {})[str(start)] = compare_rows(before['content'], after['content'])
            save()
    else:
        state['student_skipped'] = 'Teacher point estimates do not meet user greedy/sampling tolerances; adjust preservation before student training.'
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
