"""Evaluate an explicit saved model on a fixed GSM slice with the unchanged evaluator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--output',required=True)
p.add_argument('--split',choices=['train','test'],default='test')
p.add_argument('--start',type=int,default=0)
p.add_argument('--count',type=int,default=200)
p.add_argument('--max-tokens',type=int,default=512)
a=p.parse_args()
out=Path(a.output)
assert not out.exists(),f'Refusing overwrite {out}'
source=Path('/scratch/wzhao20/DOGe-official/scripts/generate-eval-qwen2_5.py')
root=Path(__file__).resolve().parents[2]
prior=json.loads((root/'results/heldout200/split_manifest.json').read_text())
digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==prior['evaluator_sha256'],'Evaluator changed since paired baselines'
spec=importlib.util.spec_from_file_location('existing_evaluator',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.load_dataset
def selected(*args,**kwargs):
    data=original(*args,**kwargs)
    chosen=data[a.split].select(range(a.start,a.start+a.count))
    if a.split=='train':
        audit=json.loads((root/'results/baseline_20260911/input_audit.json').read_text())
        assert not set(chosen['question']) & set(audit['training_questions']), 'Validation overlaps training'
    data['test']=chosen
    return data
module.load_dataset=selected
sys.argv=[str(source),'--model_path',a.model,'--output_dir',str(out),'--limit',str(a.count),
          '--batch_size','8','--max_tokens',str(a.max_tokens)]
module.main()
path=out/'gsm8k-results.json'
d=json.loads(path.read_text())
assert len(d['content'])==a.count
for row in d['content']:row['id']+=a.start
d['evaluation_split']={'dataset':'openai/gsm8k','subset':'main','split':a.split,
                       'indices':[a.start,a.start+a.count],'evaluator_sha256':digest}
path.write_text(json.dumps(d,indent=2))
