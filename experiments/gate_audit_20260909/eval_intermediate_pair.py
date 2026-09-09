"""Post-hoc dose diagnostics on already saved, matched 40/80-step checkpoints."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--step', type=int, choices=[40,80], required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
scripts = Path(__file__).resolve().parent
results = root / 'results'
source = Path('/scratch/wzhao20/DOGe-official/scripts/generate-eval-qwen2_5.py')
manifest = json.loads((results/'heldout200/split_manifest.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['evaluator_sha256']
base = json.loads((results/'corrected_numeric.json').read_text())['base_path']
paths = {}
for arm in ['clean','contrast']:
    models = list((results/f'repaired_{arm}_t120').glob(f'**/{args.step}/pytorch_model.bin'))
    assert len(models) == 1, models
    out = results/f'repaired_{arm}_step{args.step}_gsm200'
    assert not out.exists(), f'Refusing overwrite: {out}'
    subprocess.run([sys.executable,str(source),'--model_path',str(models[0].parent),
                    '--output_dir',str(out),'--limit','200','--batch_size','8',
                    '--max_tokens','512'],cwd=root,check=True)
    paths[arm] = str(out/'gsm8k-results.json')
comparison = results/f'repaired_step{args.step}_comparison.json'
subprocess.run([sys.executable,str(scripts/'compare_gate.py'),'--baseline',base,
                '--clean',paths['clean'],'--candidate',paths['contrast'],
                '--output',str(comparison)],cwd=root,check=True)
d = json.loads(comparison.read_text())
d['analysis_scope'] = ('Post-hoc intermediate checkpoint diagnostic after observing primary120 collapse; '
                       'not an independently selected final endpoint. Clean/gate update counts match.')
comparison.write_text(json.dumps(d,indent=2))
subprocess.run([sys.executable,str(scripts/'quality_diagnostics.py'),'--baseline',base,
                '--candidate',paths['contrast'],'--output',str(results/f'repaired_step{args.step}_quality.json')],
               cwd=root,check=True)
