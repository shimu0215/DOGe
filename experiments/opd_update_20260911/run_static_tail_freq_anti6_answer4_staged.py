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
tag = 'static_tail_freq_anti6_answer4_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 2400
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
    mixed_context = OUT / 'static_mixctx_top2_9870980_context'
    assert read(mixed_context/'manifest.json')['complete']
    trainer = Path(__file__).with_name('train_static_tail_last2_frequent_rl.py')
    import hashlib
    previous = read(OUT / 'static_mixctx_anchor12_9908591' / 'manifest.json')
    assert previous['complete'] and previous['teacher'] == os.environ['TEACHER']
    assert previous['code_sha256'] == hashlib.sha256(trainer.with_name('train_static_top2.py').read_bytes()).hexdigest()
    assert previous['context'] == str(mixed_context)
    assert previous['anchor_weight'] == 12 and previous['anti_weight'] == 1
    state['recipe_change'] = 'Repair failed frequentRL anti6 by finalanswerCE1to4; staticanchor24 live6 ownRL.5 last2 anti6 64 unchanged. No student signals.'
    save()
    for phase, steps in [('smoke', 2), ('train', 64)]:
        output = OUT / (tag + ('_smoke' if phase == 'smoke' else ''))
        cmd = [os.environ['PY'], str(trainer), '--teacher', os.environ['TEACHER'],
               '--context', str(mixed_context), '--output', str(output),
               '--steps', str(steps), '--rl-every', '1', '--anti-weight', '6', '--anchor-weight', '24', '--answer-weight', '4', '--rl-weight', '.5', '--warmup', '16', '--ramp-end', '48']
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
    extra_dest = OUT / (tag + '_teacher_extra200')
    run('teacher_extra200', [os.environ['PY'], str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
        '--model', str(teacher), '--examples', str(raw_examples), '--output', str(extra_dest),
        '--limit', '200', '--modes', 'greedy', 'sampling', '--max-tokens', '512'])
    assert read(extra_dest/'summary.json')['complete']
    extra_pass = True
    for mode in ['greedy', 'sampling']:
        before = [json.loads(x) for x in (OUT/'static_original_extra200_9870980'/(mode+'.jsonl')).read_text().splitlines()]
        after = [json.loads(x) for x in (extra_dest/(mode+'.jsonl')).read_text().splitlines()]
        assert len(before) == len(after) == 200
        result = compare_rows(before, after)
        state.setdefault('teacher_extra200', {})[mode] = result
        extra_pass &= result['paired']['delta_pp'] >= (-1. if mode == 'sampling' else 0.) - 1e-8
    state['extra_point_tolerance_pass'] = extra_pass
    state['extra_quality_scope'] = 'Supplementary check on previously used extra200; retain failures. Main400 admission unchanged.'
    save()
    state['student_skipped'] = 'Staged teacher quality only; no student has been trained. Independent OPD requires this full record complete and main quality pass.'
    state['staged_training_only'] = True
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
