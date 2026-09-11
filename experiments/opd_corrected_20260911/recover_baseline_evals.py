"""Evaluate saved baseline checkpoints after interruption, without resuming training."""
import argparse
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
OUT = ROOT / 'results/opd_corrected_20260911'
record = OUT / ('recovery_baseline_' + a.job + '.json')
assert not record.exists(), record
state = dict(start=time.time(), job=a.job, complete=False, completed=[], results={})

def save():
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(record)

def read(path):
    return json.loads(path.read_text())

def evaluate(label, model, split, start, count):
    target = OUT / ('recovery_' + a.job + '_' + label)
    assert not target.exists(), target
    command = [os.environ['PY'], str(ROOT / 'experiments/baseline_20260911/evaluate.py'),
               '--model', str(model), '--output', str(target), '--split', split,
               '--start', str(start), '--count', str(count)]
    state.update(phase=label, command=command)
    save()
    with target.with_suffix('.log').open('x') as log:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    state['completed'].append(label)
    save()
    return read(target / 'gsm8k-results.json')

try:
    save()
    assert os.environ['SLURM_JOB_ID'] == a.job
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()
    import torch
    assert torch.cuda.device_count() == 1
    state['device'] = dict(visible=os.environ['CUDA_VISIBLE_DEVICES'],
                          step=os.environ['SLURM_STEP_ID'],
                          uuid=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                                        '--format=csv,noheader'], text=True).strip())
    sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction, gold, paired

    def compare(before, after):
        key = lambda d: [(r['id'], r['prompt'], r['ground_truth']) for r in d['content']]
        assert key(before) == key(after) and before['generation'] == after['generation']
        def scores(d):
            return [int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth']))
                    for r in d['content']]
        x, y = scores(before), scores(after)
        return dict(initial=sum(x)/len(x), accuracy=sum(y)/len(y), paired=paired(x, y))

    initial = read(ROOT / 'results/baseline_20260911/full_sft_val/gsm8k-results.json')
    for mode in ['minillm_highlr', 'reverse_kl', 'immediate_pg']:
        candidates = []
        result = state['results'][mode] = dict(validation={})
        for step in [120, 240, 360, 480]:
            paths = list((OUT / (mode + '_clean480_s10_opd')).glob('**/' + str(step) + '/pytorch_model.bin'))
            if not paths:
                continue
            assert len(paths) == 1
            model = paths[0].parent
            data = evaluate(mode + '_val' + str(step), model, 'train', 7000, 128)
            comp = compare(initial, data)
            result['validation'][str(step)] = comp
            candidates.append((comp['paired']['delta_pp'], step, model))
            save()
        if not candidates:
            result['skipped'] = 'No saved checkpoint; no training or resume claim'
            save()
            continue
        chosen = max(candidates, key=lambda x: (x[0], -x[1]))
        result['selected'] = dict(gain_pp=chosen[0], steps=chosen[1], model=str(chosen[2]),
                                 criterion='Best heldout-train validation among available saved checkpoints; incomplete 480-step run')
        data = evaluate(mode + '_selected_test0', chosen[2], 'test', 0, 200)
        result['old_test'] = compare(read(Path(os.environ['EXAMPLES'])), data)
        save()
        if chosen[0] >= 3:
            data = evaluate(mode + '_selected_test1000', chosen[2], 'test', 1000, 200)
            result['new_test'] = compare(read(ROOT / 'results/opd_update_20260911/full_sft_test1000_reference/gsm8k-results.json'), data)
            save()
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
