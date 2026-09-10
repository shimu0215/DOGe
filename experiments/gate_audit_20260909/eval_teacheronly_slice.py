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
p.add_argument('--start',type=int,default=400)
p.add_argument('--count',type=int,default=200)
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
    data['test']=data['test'].select(range(a.start,a.start+a.count))
    return data
module.load_dataset=selected
sys.argv=[str(source),'--model_path',a.model,'--output_dir',str(out),'--limit',str(a.count),
          '--batch_size','8','--max_tokens','512']
module.main()
path=out/'gsm8k-results.json'
d=json.loads(path.read_text())
assert len(d['content'])==a.count
for row in d['content']:row['id']+=a.start
d['evaluation_split']={'dataset':'openai/gsm8k','subset':'main','split':'test',
                       'indices':[a.start,a.start+a.count],'evaluator_sha256':digest}
path.write_text(json.dumps(d,indent=2))
