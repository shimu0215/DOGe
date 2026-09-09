"""Finalize the fixed 120-step repaired pair without changing training artifacts."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
scripts = Path(__file__).resolve().parent
results = root / 'results'
outputs = Path('/scratch/wzhao20/DOGe-official/outputs')
clean = results / 'repaired_clean_gsm200/gsm8k-results.json'
candidate = results / 'repaired_contrast_gsm200/gsm8k-results.json'
for path in [clean, candidate, results / 'teacher_fp16_preservation.json']:
    assert path.is_file(), f'Experiment not finished: {path}'
snapshot = json.loads((results / 'repaired_code_snapshot.json').read_text())
for name, expected in snapshot['files'].items():
    path = Path(name) if Path(name).is_absolute() else root / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name


def run(script, *args):
    subprocess.run([sys.executable, str(scripts / script), *map(str, args)],
                   cwd=root, check=True)


run('corrected_numeric_audit.py', '--root', outputs, '--audit-root', results)
base = json.loads((results / 'corrected_numeric.json').read_text())['base_path']
run('compare_gate.py', '--baseline', base, '--clean', clean, '--candidate', candidate,
    '--output', results / 'repaired_pair_comparison.json')
run('quality_diagnostics.py', '--baseline', base, '--candidate', candidate,
    '--output', results / 'repaired_contrast_quality.json')
manifest = {
    'analysis_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                text=True).strip(),
    'training_code_snapshot': 'results/repaired_code_snapshot.json',
    'training_files_verified': len(snapshot['files']),
    'comparison': 'results/repaired_pair_comparison.json',
    'quality': 'results/repaired_contrast_quality.json',
    'teacher_precision_matched_check': 'results/teacher_fp16_preservation.json',
    'analysis_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [scripts / x for x in ['corrected_numeric_audit.py',
                                  'compare_gate.py', 'quality_diagnostics.py']]},
    'scope': 'One training seed combination; 120 fixed updates; paired GSM test first 200. '
             'Historical checkpoint selection used this test set; no independent confirmatory claim.'
}
(results / 'repaired_experiment_manifest.json').write_text(json.dumps(manifest, indent=2))
print('FINALIZED repaired pair', flush=True)
