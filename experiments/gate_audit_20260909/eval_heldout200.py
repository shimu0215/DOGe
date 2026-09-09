"""Use the existing evaluator unchanged on GSM test indices 200:400."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['sft', 'clean', 'contrast'], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
results = root / 'results'
base = json.loads((results / 'corrected_numeric.json').read_text())['base_path']
if args.variant == 'sft':
    model = json.loads(Path(base).read_text())['model_name']
else:
    paths = list((results / f'repaired_{args.variant}_t120').glob('**/120/pytorch_model.bin'))
    assert len(paths) == 1, paths
    model = str(paths[0].parent)
output = results / 'heldout200' / args.variant
assert not output.exists(), f'Refusing overwrite: {output}'
source = Path('/scratch/wzhao20/DOGe-official/scripts/generate-eval-qwen2_5.py')
digest = hashlib.sha256(source.read_bytes()).hexdigest()
manifest_path = results / 'heldout200' / 'split_manifest.json'
manifest = {'dataset': 'openai/gsm8k', 'subset': 'main', 'split': 'test',
            'indices': [200, 400], 'evaluator_sha256': digest,
            'purpose': 'Additional questions not used in this audit for gate selection; '
                       'fixed primary seed models and 120 updates.'}
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest
else:
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

spec = importlib.util.spec_from_file_location('existing_evaluator', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_load = module.load_dataset


def load_slice(*a, **kw):
    dataset = original_load(*a, **kw)
    assert len(dataset['test']) >= 400
    dataset['test'] = dataset['test'].select(range(200, 400))
    return dataset


module.load_dataset = load_slice
sys.argv = [str(source), '--model_path', model, '--output_dir', str(output),
            '--limit', '200', '--batch_size', '8', '--max_tokens', '512']
module.main()
path = output / 'gsm8k-results.json'
data = json.loads(path.read_text())
assert len(data['content']) == 200
for row in data['content']:
    row['id'] += 200
data['evaluation_split'] = manifest
path.write_text(json.dumps(data, indent=2))
