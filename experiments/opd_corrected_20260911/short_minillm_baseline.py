"""Queue corrected MiniLLM from short SFT after recovery baseline evaluations."""
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
p.add_argument('--worker', action='store_true')
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_corrected_20260911'
tag = 'short_minillm_' + a.job
record = OUT / (tag + ('_worker' if a.worker else '_queue') + '.json')
assert not record.exists(), record
state = dict(start=time.time(), job=a.job, complete=False, completed=[])

def read(path):
    return json.loads(path.read_text())

def save():
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(record)

def run(phase, command, env=None):
    state.update(phase=phase, command=command)
    save()
    with (OUT / (tag + '_' + phase + '.log')).open('x') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        save()
        code = child.wait()
    if code:
        raise RuntimeError(phase + ' exit=' + str(code))
    state['completed'].append(phase)
    save()

def evaluate(label, model, split='train', start=7000, count=128):
    target = OUT / (tag + '_' + label)
    assert not target.exists(), target
    run(label, [os.environ['PY'], str(ROOT / 'experiments/baseline_20260911/evaluate.py'),
                '--model', str(model), '--output', str(target), '--split', split,
                '--start', str(start), '--count', str(count)])
    return read(target / 'gsm8k-results.json')

try:
    save()
    if not a.worker:
        state['phase'] = 'waiting_recovery_complete'
        save()
        while True:
            previous = read(OUT / ('recovery_baseline_' + a.job + '.json'))
            if previous.get('error'):
                raise RuntimeError('Recovery predecessor failed; inspect before reuse')
            if previous.get('complete'):
                break
            time.sleep(20)
        info = subprocess.check_output(['scontrol', 'show', 'job', a.job, '-o'], text=True)
        fields = dict(x.split('=', 1) for x in info.split() if '=' in x)
        tres = dict(x.split('=', 1) for x in fields['AllocTRES'].split(','))
        assert fields['JobState'] == 'RUNNING' and tres['gres/gpu'] == '1'
        remaining = datetime.datetime.fromisoformat(fields['EndTime']).timestamp() - time.time()
        assert remaining >= 6000, ('Insufficient complete-pipeline budget', remaining)
        # Wait until the predecessor srun step exits, not just its final JSON write.
        while True:
            steps = subprocess.check_output(['squeue', '--steps', '-j', a.job, '-h', '-o', '%i'], text=True).split()
            active = [s for s in steps if s.rsplit('.', 1)[-1] not in ['batch', 'extern']]
            if not active:
                break
            time.sleep(20)
        state['allocation'] = info
        save()
        run('allocated_step', ['srun', '--jobid=' + a.job, '--overlap', '--exact', '--cpu-bind=none',
                               '-N1', '-n1', '-c4', '--gres=gpu:1', os.environ['PY'],
                               str(Path(__file__)), '--job', a.job, '--worker'])
    else:
        assert os.environ['SLURM_JOB_ID'] == a.job
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        import torch
        assert torch.cuda.device_count() == 1
        state['device'] = dict(visible=os.environ['CUDA_VISIBLE_DEVICES'], step=os.environ['SLURM_STEP_ID'],
                              uuid=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip())
        student = ROOT / 'results/baseline_20260911/short_sft/checkpoint-49'
        state['protocol'] = dict(student=str(student), teacher=os.environ['TEACHER'], objective='corrected_minillm',
                                 seed=10, lr=1e-6, actual_updates=240, selection='Heldout train128; lower step breaks ties')
        save()
        run('cpu_objectives', [os.environ['PY'], str(Path(__file__).parent / 'check_objectives.py')])
        env = os.environ.copy()
        label = tag + '_clean240_s10'
        env.update(INTERNAL_TEACHER=env['TEACHER'], INTERNAL_STUDENT=str(student), INTERNAL_LABEL=label,
                   INTERNAL_SEED='10', INTERNAL_PPO_SEED='42', INTERNAL_LM_SEED='7', INTERNAL_PORT='30515',
                   BASELINE_STEPS='240', BASELINE_LR='1e-6', CORRECTED_MODE='minillm',
                   CORRECTED_RECORD=str(OUT / (label + '_updates.json')),
                   CORRECTED_STUDENT_VOCAB=str(read(student / 'config.json')['vocab_size']), CORRECTED_SAVE_INTERVAL='120')
        run('train240', ['bash', str(Path(__file__).parent / 'opd.sh')], env)
        updates = read(OUT / (label + '_updates.json'))
        assert updates['complete'] and updates['actual_optimizer_steps'] == 240
        assert updates['teacher_dtype'] == 'torch.float16' and updates['student_dtype'] == 'torch.bfloat16'
        assert updates['updates'][-1]['master_delta_rms'] > 0
        sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
        from corrected_numeric_audit import prediction, gold, paired

        def compare(before, after):
            key = lambda d: [(r['id'], r['prompt'], r['ground_truth']) for r in d['content']]
            assert key(before) == key(after) and before['generation'] == after['generation']
            def scores(d):
                return [int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth'])) for r in d['content']]
            x, y = scores(before), scores(after)
            return dict(initial=sum(x)/len(x), accuracy=sum(y)/len(y), paired=paired(x, y))

        initial = read(ROOT / 'results/baseline_20260911/short_checkpoint-49_val/gsm8k-results.json')
        candidates = []
        for step in [120, 240]:
            paths = list((OUT / (label + '_opd')).glob('**/' + str(step) + '/pytorch_model.bin'))
            assert len(paths) == 1
            model = paths[0].parent
            comp = compare(initial, evaluate('val' + str(step), model))
            state.setdefault('validation', {})[str(step)] = comp
            candidates.append((comp['paired']['delta_pp'], step, model))
            save()
        chosen = max(candidates, key=lambda c: (c[0], -c[1]))
        state['selected'] = dict(gain_pp=chosen[0], step=chosen[1], model=str(chosen[2]))
        save()
        for start, initial_path in [(0, OUT / 'short_fkl_initial_test0/gsm8k-results.json'),
                                    (1000, ROOT / 'results/baseline_20260911/short_initial_test1000/gsm8k-results.json')]:
            initial_test = read(initial_path) if initial_path.exists() else evaluate('initial_test' + str(start), student, 'test', start, 200)
            data = evaluate('selected_test' + str(start), chosen[2], 'test', start, 200)
            state.setdefault('test', {})[str(start)] = compare(initial_test, data)
            save()
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
