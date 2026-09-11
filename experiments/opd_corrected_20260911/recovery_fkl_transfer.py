"""Fixed plain FKL teacher: preservation and transfer to corrected MiniLLM.

Run only inside a verified allocated one-GPU srun step. All outputs are new.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--job', required=True)
p.add_argument('--port', type=int, default=30513)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_corrected_20260911'
SCRIPTS = Path(__file__).parent
tag = 'recovery_fkl_transfer_' + a.job
record = OUT / (tag + '.json')
assert not record.exists(), record
state = dict(start=time.time(), job=a.job, complete=False, completed=[])

def save():
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(record)

def read(path):
    return json.loads(path.read_text())

def run(phase, command, env=None):
    state.update(phase=phase, command=command)
    save()
    with (OUT / (tag + '_' + phase + '.log')).open('x') as log:
        subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    state['completed'].append(phase)
    save()

def train(steps):
    label = tag + ('_smoke4' if steps == 4 else '_defense240_s10')
    env = os.environ.copy()
    env.update(INTERNAL_TEACHER=str(teacher), INTERNAL_STUDENT=env['PROXY'],
               INTERNAL_LABEL=label, INTERNAL_SEED='10', INTERNAL_PPO_SEED='42',
               INTERNAL_LM_SEED='7', INTERNAL_PORT=str(a.port), BASELINE_STEPS=str(steps),
               BASELINE_LR='1e-6', CORRECTED_MODE='minillm',
               CORRECTED_RECORD=str(OUT / (label + '_updates.json')),
               CORRECTED_STUDENT_VOCAB=str(read(Path(env['PROXY']) / 'config.json')['vocab_size']),
               CORRECTED_SAVE_INTERVAL='4' if steps == 4 else '120')
    run(label, ['bash', str(SCRIPTS / 'opd.sh')], env)
    d = read(OUT / (label + '_updates.json'))
    assert d['complete'] and d['actual_optimizer_steps'] == steps
    assert d['teacher_dtype'] == 'torch.float16' and d['student_dtype'] == 'torch.bfloat16'
    assert d['updates'][-1]['master_delta_rms'] > 0
    paths = list((OUT / (label + '_opd')).glob('**/' + str(steps) + '/pytorch_model.bin'))
    assert len(paths) == 1
    return paths[0].parent

def evaluate(label, model, start):
    dest = OUT / (tag + '_' + label)
    assert not dest.exists()
    run(label, [os.environ['PY'], str(ROOT / 'experiments/baseline_20260911/evaluate.py'),
                '--model', str(model), '--output', str(dest), '--split', 'test',
                '--start', str(start), '--count', '200'])
    return read(dest / 'gsm8k-results.json')

try:
    save()
    assert os.environ['SLURM_JOB_ID'] == a.job
    info = subprocess.check_output(['scontrol', 'show', 'job', a.job, '-o'], text=True)
    fields = dict(x.split('=', 1) for x in info.split() if '=' in x)
    assert fields['JobState'] == 'RUNNING'
    remaining = datetime.datetime.fromisoformat(fields['EndTime']).timestamp() - time.time()
    assert remaining >= 9000, ('Insufficient complete-pipeline budget', remaining)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()
    import torch
    assert torch.cuda.device_count() == 1
    state['allocation'] = info
    state['device'] = dict(visible=os.environ['CUDA_VISIBLE_DEVICES'], step=os.environ['SLURM_STEP_ID'],
                          uuid=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                                        '--format=csv,noheader'], text=True).strip())
    teacher = ROOT / 'results/opd_update_20260911/direct_fkl/model'
    manifest = read(teacher.parent / 'manifest.json')
    assert manifest['complete'] and manifest['plain_export_verified']
    hashes = {path: r['sha256'] for path, r in read(ROOT / 'results/repaired_code_snapshot.json')['files'].items()
              if Path(path).is_absolute()}
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    state['shared_sha256'] = hashes
    sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction, gold, paired

    def scores(rows):
        return [int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth'])) for r in rows]

    def compare_rows(before, after):
        key = lambda rows: [(r['id'], r['prompt'], r['ground_truth']) for r in rows]
        assert key(before) == key(after)
        x, y = scores(before), scores(after)
        return dict(initial=sum(x)/len(x), candidate=sum(y)/len(y), paired=paired(x, y))

    def compare(before, after):
        assert before['generation'] == after['generation']
        return compare_rows(before['content'], after['content'])

    examples = ROOT / 'results/baseline_20260911/short_initial_test1000/gsm8k-results.json'
    teacher_outputs = {}
    for name, model in [('original', os.environ['TEACHER']), ('defense', teacher)]:
        dest = OUT / (tag + '_' + name + '_teacher_new200')
        run(name + '_teacher_new200', [os.environ['PY'], str(ROOT / 'experiments/internalize_20260910/evaluate_plain.py'),
                                    '--model', str(model), '--examples', str(examples), '--output', str(dest),
                                    '--limit', '200', '--modes', 'greedy', 'sampling'])
        assert read(dest / 'summary.json')['complete']
        teacher_outputs[name] = dest
    state['teacher_new200'] = {}
    for mode in ['greedy', 'sampling']:
        rows = [[json.loads(line) for line in (teacher_outputs[name] / (mode + '.jsonl')).read_text().splitlines()]
                for name in ['original', 'defense']]
        state['teacher_new200'][mode] = compare_rows(*rows)
    state['limitation'] = 'Transfer test proceeds as diagnostic even if teacher sampling declines; no joint success claim.'
    save()
    run('cpu_objectives', [os.environ['PY'], str(SCRIPTS / 'check_objectives.py')])
    train(4)
    final = train(240)
    for start in [0, 1000]:
        initial = read(Path(os.environ['EXAMPLES'])) if start == 0 else read(ROOT / 'results/opd_update_20260911/full_sft_test1000_reference/gsm8k-results.json')
        for step in [120, 240]:
            paths = list((OUT / 'minillm_clean240_s10_opd').glob('**/' + str(step) + '/pytorch_model.bin'))
            assert len(paths) == 1
            clean_path = OUT / ('rank_minillm_clean_test0_' + str(step)) / 'gsm8k-results.json'
            clean = read(clean_path) if start == 0 and clean_path.exists() else evaluate('clean_test' + str(start) + '_' + str(step), paths[0].parent, start)
            candidate = evaluate('defense_test' + str(start) + '_' + str(step), final.parent / str(step), start)
            state.setdefault('student_results', {})[str(start) + '_' + str(step)] = dict(vs_clean=compare(clean, candidate), vs_sft=compare(initial, candidate))
            save()
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
