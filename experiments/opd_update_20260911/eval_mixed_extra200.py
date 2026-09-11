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
p.add_argument('--variant', choices=['mixed'], required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = 'static_' + a.variant + '_extra200_' + a.job
source_tag = {'mixed': 'static_mixctx_top2_9870980'}[a.variant]
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 1800
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
    import hashlib
    import importlib.util
    from datasets import load_dataset
    from transformers import AutoTokenizer
    teacher = Path(os.environ['TEACHER']) if a.variant == 'original' else OUT / source_tag / 'model'
    if a.variant != 'original':
        provenance = read(teacher.parent / 'manifest.json')
        assert provenance['complete'] and provenance['plain_export_verified']
        assert provenance['student_model_loaded'] is False
        assert provenance['student_parameter_signal'] is False
        assert provenance['student_outcome_reward'] is False
        assert provenance['teacher'] == os.environ['TEACHER']
    evaluator = Path('/scratch/wzhao20/DOGe-official/scripts/generate-eval-qwen2_5.py')
    expected = read(ROOT / 'results/heldout200/split_manifest.json')['evaluator_sha256']
    assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('original_evaluator', evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = load_dataset('openai/gsm8k', 'main')['test']
    tokenizer = AutoTokenizer.from_pretrained(os.environ['TEACHER'])
    def make_row(i):
        row = dataset[i]
        prompt = tokenizer.apply_chat_template([
            dict(role='system', content=module.SYSTEM_PROMPT),
            dict(role='user', content=row['question'])], tokenize=False, add_generation_prompt=True)
        return dict(id=i, prompt=prompt, ground_truth=row['answer'])
    rows = [make_row(i) for i in range(200, 400)]
    prior = [make_row(i) for i in list(range(200)) + list(range(1000, 1200))]
    assert not set(r['prompt'] for r in rows) & set(r['prompt'] for r in prior)
    examples = OUT / (tag + '_examples.json')
    with examples.open('x') as handle:
        json.dump(dict(content=rows), handle, indent=2)
    state.update(teacher_source=str(teacher), teacher_training_performed=False,
        split=dict(dataset='openai/gsm8k', subset='main', split='test', start=200, count=200,
                   disjoint_from_current400=True, historical_use_not_audited=True,
                   examples_sha256=hashlib.sha256(examples.read_bytes()).hexdigest()),
        purpose='Additional fixed-checkpoint quality check for mixed-context teacher; original per-slice screen passed')
    destination = OUT / tag
    env = {k:v for k,v in os.environ.items() if k != 'PROXY' and not k.startswith('INTERNAL_')}
    run('teacher_extra200', [os.environ['PY'], str(ROOT / 'experiments/internalize_20260910/evaluate_plain.py'),
        '--model', str(teacher), '--examples', str(examples), '--output', str(destination),
        '--limit', '200', '--modes', 'greedy', 'sampling', '--max-tokens', '512'], env)
    sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction, gold
    state['scores'] = {}
    for mode in ['greedy', 'sampling']:
        outputs = [json.loads(line) for line in (destination / (mode + '.jsonl')).read_text().splitlines()]
        assert [(r['id'],r['prompt'],r['ground_truth']) for r in outputs] == [(r['id'],r['prompt'],r['ground_truth']) for r in rows]
        correct = [int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth'])) for r in outputs]
        state['scores'][mode] = dict(correct=sum(correct), n=len(correct), accuracy=sum(correct)/len(correct))
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
