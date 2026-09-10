"""Paired 1024-token diagnostic, without changing the 512-token primary endpoint."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from transformers import AutoTokenizer
from corrected_numeric_audit import prediction, gold
from compare_gate import compare

root = Path(__file__).resolve().parents[2]
scripts = Path(__file__).resolve().parent
os.chdir(root)
out = root/'results/teacheronly_research/length1024'
out.mkdir(parents=True, exist_ok=False)
sources = {
    'sft': Path(json.loads((root/'results/corrected_numeric.json').read_text())['base_path']),
    'clean_s10': root/'results/repaired_clean_gsm200/gsm8k-results.json',
    'top32_s10': root/'results/teacheronly_top32_s10_gsm200/gsm8k-results.json',
}
training = json.loads((root/'results/teacheronly_research/teacheronly_top32_s10/manifest.json').read_text())
assert training.get('complete') and training.get('code_verified')
results = {}
for label, source in sources.items():
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = json.loads(source.read_text())['model_name']
    record = out/f'{label}.manifest.json'
    manifest = {'label': label, 'source': str(source), 'source_sha256': digest,
                'model': model, 'indices': [0, 200], 'max_tokens': 1024,
                'purpose': 'Post-selection length diagnostic, not a new selection endpoint',
                'start': time.time(), 'complete': False}
    record.write_text(json.dumps(manifest, indent=2))
    print(f'START length1024 {label}', flush=True)
    with (out/f'{label}.log').open('w') as log:
        subprocess.run([sys.executable, str(scripts/'eval_teacheronly_slice.py'),
                        '--model', model, '--output', str(out/label), '--start', '0',
                        '--max-tokens', '1024'], stdout=log, stderr=subprocess.STDOUT, check=True)
    d = json.loads((out/label/'gsm8k-results.json').read_text())
    assert d['evaluation_split']['indices'] == [0, 200]
    assert d['model_name'] == model
    assert [r['id'] for r in d['content']] == list(range(200))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    results[label] = d
    manifest.update(end=time.time(), complete=True)
    record.write_text(json.dumps(manifest, indent=2))
    print(f'COMPLETE length1024 {label}', flush=True)
tokenizer = AutoTokenizer.from_pretrained(results['sft']['model_name'])
scores = {}
summary = {'n': 200, 'indices': [0, 200], 'max_tokens': 1024, 'models': {}}
for label, d in results.items():
    assert d['generation'] == results['sft']['generation']
    assert [(r['prompt'], r['ground_truth']) for r in d['content']] == [(r['prompt'], r['ground_truth']) for r in results['sft']['content']]
    parsed = [prediction(r['prediction']) for r in d['content']]
    scores[label] = np.array([int(v[0] == gold(r['ground_truth'])) for v, r in zip(parsed, d['content'])])
    lengths = [len(tokenizer.encode(r['prediction'], add_special_tokens=False)) for r in d['content']]
    summary['models'][label] = {'accuracy': float(scores[label].mean()),
                              'mean_reencoded_tokens': float(np.mean(lengths)),
                              'near1024_token_cap_proxy': float(np.mean(np.array(lengths) >= 1022)),
                              'complete_numeric_boxes': sum(value is not None and method == 'boxed' for value, method in parsed)}
summary['top32_vs_sft'] = compare(scores['sft'], scores['top32_s10'])
summary['top32_vs_clean'] = compare(scores['clean_s10'], scores['top32_s10'])
(out/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
