"""Finalize the fixed 120-step repaired pair without changing training artifacts."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
scripts = Path(__file__).resolve().parent
results = root / 'results'
outputs = Path('/scratch/wzhao20/DOGe-official/outputs')
parser = argparse.ArgumentParser()
parser.add_argument('--replicate', choices=['primary', 'seed11'], default='primary')
args = parser.parse_args()
suffix = '_seed11' if args.replicate == 'seed11' else ''
prefix = 'repaired' + suffix
clean = results / f'repaired_clean{suffix}_gsm200/gsm8k-results.json'
candidate = results / f'repaired_contrast{suffix}_gsm200/gsm8k-results.json'
for path in [clean, candidate, results / 'teacher_fp16_preservation.json']:
    assert path.is_file(), f'Experiment not finished: {path}'
snapshot_name = f'{prefix}_code_snapshot.json'
snapshot = json.loads((results / snapshot_name).read_text())
for name, expected in snapshot['files'].items():
    path = Path(name) if Path(name).is_absolute() else root / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name


def run(script, *args):
    subprocess.run([sys.executable, str(scripts / script), *map(str, args)],
                   cwd=root, check=True)


run('corrected_numeric_audit.py', '--root', outputs, '--audit-root', results)
base = json.loads((results / 'corrected_numeric.json').read_text())['base_path']
run('compare_gate.py', '--baseline', base, '--clean', clean, '--candidate', candidate,
    '--output', results / f'{prefix}_pair_comparison.json')
run('quality_diagnostics.py', '--baseline', base, '--candidate', candidate,
    '--output', results / f'{prefix}_contrast_quality.json')
manifest = {
    'analysis_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                text=True).strip(),
    'replicate': args.replicate,
    'seeds': [11,43,8] if suffix else [10,42,7],
    'training_code_snapshot': f'results/{snapshot_name}',
    'training_files_verified': len(snapshot['files']),
    'comparison': f'results/{prefix}_pair_comparison.json',
    'quality': f'results/{prefix}_contrast_quality.json',
    'teacher_precision_matched_check': 'results/teacher_fp16_preservation.json',
    'analysis_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [scripts / x for x in ['corrected_numeric_audit.py',
                                  'compare_gate.py', 'quality_diagnostics.py']]},
    'scope': 'One training seed combination; 120 fixed updates; paired GSM test first 200. '
             'Historical checkpoint selection used this test set; no independent confirmatory claim.'
}
(results / f'{prefix}_experiment_manifest.json').write_text(json.dumps(manifest, indent=2))
print('FINALIZED repaired pair', args.replicate, flush=True)
