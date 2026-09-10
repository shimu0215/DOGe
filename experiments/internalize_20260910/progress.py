"""Compact read-only research progress for continuation; no GPU interaction."""
import json
from pathlib import Path
import re
import time

root=Path(__file__).resolve().parents[2]/'results/internalize'
result={'time':time.time(),'teachers':{},'evaluation':{},'opd':{},'logs':{}}
for path in sorted(root.glob('*/manifest.json')):
    data=json.loads(path.read_text())
    if 'scope' in data:
        row={k:data.get(k) for k in ['scope','modifier','lr','preserve','epochs','complete','trainable_parameters','steps']}
        curve=path.parent/'validation.jsonl'
        if curve.exists():row['validation']=json.loads(curve.read_text().splitlines()[-1])
        result['teachers'][path.parent.name]=row
for path in sorted(root.glob('*/summary.json')):
    data=json.loads(path.read_text())
    if 'modes' in data:
        result['evaluation'][path.parent.name]={'complete':data.get('complete',False),
            'n':data['n'],'modes':{mode:{key:row.get(key) for key in ['accuracy','cap_rate','unparsed_ids']}
                                   for mode,row in data['modes'].items()}}
for path in sorted(root.glob('*_opd_manifest.json')):
    data=json.loads(path.read_text())
    result['opd'][path.stem]={k:data.get(k) for k in ['complete','error','code_verified','start','end']}
for path in sorted(root.glob('*.log')):
    lines=path.read_text(errors='replace').splitlines()
    progress=[line for line in lines if re.search(r'global iter:|^generated |^greedy |^sampling |^TRAIN |^VALIDATION |^COMPLETE|^scored ',line)]
    errors=[line for line in lines if re.search(r'Error:|Traceback|OutOfMemory|FAILED',line)]
    result['logs'][path.name]={'last':progress[-1][:800] if progress else (lines[-1][:200] if lines else ''),
                             'errors':errors[-3:]}
for path in sorted(root.glob('*_comparison.json')):
    result.setdefault('comparisons',{})[path.stem]=json.loads(path.read_text())
print(json.dumps(result,indent=2))
