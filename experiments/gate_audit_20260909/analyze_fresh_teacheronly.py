"""Paired confirmation and output diagnostics after fixed-slice evaluations finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['top32', 'uniform', 'prompt', 'digits'], required=True)
p.add_argument('--seed', type=int, choices=[10, 11], required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
scripts = Path(__file__).resolve().parent
fresh = root/'results/teacheronly_research/fresh400'
labels = {'baseline': 'sft', 'clean': f'clean_s{a.seed}', 'candidate': f'{a.variant}_s{a.seed}'}
for label in labels.values():
    manifest = json.loads((fresh/f'{label}.manifest.json').read_text())
    assert manifest.get('complete') and manifest['indices'] == [400, 600], label
out = fresh/f'analysis_{a.variant}_s{a.seed}'
out.mkdir(exist_ok=False)
paths = {key: str(fresh/label/'gsm8k-results.json') for key, label in labels.items()}
for script, name in [('compare_gate.py', 'comparison'), ('quality_diagnostics.py', 'quality'),
                     ('teacheronly_behavior.py', 'behavior')]:
    args = ['--baseline', paths['baseline'], '--candidate', paths['candidate'],
            '--output', str(out/f'{name}.json')]
    if name == 'comparison':
        args.extend(['--clean', paths['clean']])
    with (out/f'{name}.log').open('w') as log:
        subprocess.run([sys.executable, str(scripts/script), *args],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
comparison = json.loads((out/'comparison.json').read_text())
print(json.dumps({'variant': a.variant, 'seed': a.seed,
                  'accuracy': {key: m['accuracy'] for key, m in comparison['models'].items()},
                  'vs_sft': comparison['candidate_vs_sft'],
                  'vs_clean': comparison['candidate_vs_clean_opd']}, indent=2))
