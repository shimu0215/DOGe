"""Evaluate fixed models on the previously reserved 400:600 confirmation slice."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--labels', nargs='+', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
scripts = Path(__file__).resolve().parent
os.chdir(root)
sources = {
    'sft': Path(json.loads((root/'results/corrected_numeric.json').read_text())['base_path']),
    'clean_s10': root/'results/repaired_clean_gsm200/gsm8k-results.json',
    'clean_s11': root/'results/repaired_clean_seed11_gsm200/gsm8k-results.json',
}
for variant in ('digits', 'top32', 'uniform', 'prompt'):
    for seed in (10, 11):
        sources[f'{variant}_s{seed}'] = root/f'results/teacheronly_{variant}_s{seed}_gsm200/gsm8k-results.json'
assert len(set(a.labels)) == len(a.labels)
assert all(label in sources for label in a.labels), a.labels
out = root/'results/teacheronly_research/fresh400'
out.mkdir(parents=True, exist_ok=True)
for label in a.labels:
    source = sources[label]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = json.loads(source.read_text())['model_name']
    if label not in ('sft', 'clean_s10', 'clean_s11'):
        training = json.loads((root/f'results/teacheronly_research/teacheronly_{label}/manifest.json').read_text())
        assert training.get('complete') and training.get('code_verified'), label
    record = out/f'{label}.manifest.json'
    destination = out/label
    assert not record.exists() and not destination.exists(), f'Refusing overwrite: {label}'
    manifest = {'label': label, 'source': str(source), 'source_sha256': digest,
                'model': model, 'indices': [400, 600], 'start': time.time(), 'complete': False}
    record.write_text(json.dumps(manifest, indent=2))
    print(f'START fresh 400:600 {label}', flush=True)
    with (out/f'{label}.log').open('w') as log:
        subprocess.run([sys.executable, str(scripts/'eval_teacheronly_slice.py'),
                        '--model', model, '--output', str(destination)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((destination/'gsm8k-results.json').read_text())
    assert result['evaluation_split']['indices'] == [400, 600]
    assert result['model_name'] == model
    assert [row['id'] for row in result['content']] == list(range(400, 600))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    manifest.update(end=time.time(), complete=True)
    record.write_text(json.dumps(manifest, indent=2))
    print(f'COMPLETE fresh 400:600 {label}', flush=True)
