"""Symbolic mathematics scoring with isolated dependencies, not numeric-tail GSM scoring."""
import argparse,json,sys,importlib.metadata
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'results/generalization_20260913/math_verify_deps'))
from math_verify import parse,verify,LatexExtractionConfig
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--check',action='store_true');a=p.parse_args()
def gold(s):return parse('$'+s+'$',extraction_config=[LatexExtractionConfig()])
if a.check:
 for pred,target,want in [(r'\boxed{\frac12}',r'0.5',True),(r'\boxed{x+x}',r'2x',True),(r'\boxed{\sqrt{4}}','2',True),(r'\boxed{2}','3',False),(r'\boxed{(1,2)}','(1,2)',True)]:
  g=gold(target);v=parse(pred);assert g and v and bool(verify(g,v))==want,(pred,target)
 data=json.loads((root/'AgentDistill/data_processor/math_dataset/test/math_500_20250414.json').read_text())['examples'][:100]
 assert all(gold(r['answer']) for r in data)
 print('PASS symbolic/fraction/tuple checks and100gold parsing');sys.exit(0)
pth=Path(a.input);d=json.loads(pth.read_text());assert d['complete'] and len(d['content'])==100
for r in d['content']:
 g=gold(r['answer']);v=parse(r['prediction']);assert g
 r.update(correct=bool(verify(g,v)),prediction_parsed=bool(v))
d['correct']=sum(r['correct'] for r in d['content']);d['accuracy']=d['correct']/100;d['scorer']=dict(name='math-verify',version=importlib.metadata.version('math-verify'),isolated_dependencies=True)
(pth.parent/'scored.json').write_text(json.dumps(d,indent=2));print(d['correct'])
