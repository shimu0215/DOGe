"""Audit a narrowly specified LaTeX thin-space parsing fix across all controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired

def corrected(text):
    # The original numeric() removes commas before removing '\,', leaving '\'.
    # A LaTeX thin space carries no numerical meaning. Normalize it first.
    return prediction(text.replace(r'\,',' '))

assert corrected(r'\boxed{230 \, \text{miles}}')[0]==230
assert corrected(r'\boxed{1,200 \, \text{dollars}}')[0]==1200
assert corrected(r'\boxed{\frac{1,000}{2}}')[0]==500
assert corrected(r'\boxed{230 \, \text{miles}} revised \boxed{7}')[0]==7
assert corrected(r'\boxed{x \, + 230}')[0] is None
assert corrected(r'\boxed{\text{No savings}}')[0] is None

p=argparse.ArgumentParser();p.add_argument('--root',required=True)
p.add_argument('--baseline',required=True);p.add_argument('--clean',required=True)
p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root)
result={'rule':'Remove literal LaTeX thin-space command before existing numeric parser; every other rule unchanged',
    'generation_and_training_unchanged':True,'records':{},'paired_students':{}}
cache={}
def audit(key,rows):
    old=[];new=[];changes=[]
    for r in rows:
        before,method=prediction(r['prediction']);after,_=corrected(r['prediction'])
        truth=gold(r['ground_truth'])
        old.append(int(before is not None and before==truth))
        new.append(int(after is not None and after==truth))
        if before!=after:
            changes.append({'id':r['id'],'old':str(before),'new':str(after),'gold':str(truth),
                'old_correct':bool(old[-1]),'new_correct':bool(new[-1]),
                'prediction':r['prediction']})
    result['records'][key]={'n':len(rows),'old_accuracy':sum(old)/len(old),
        'corrected_accuracy':sum(new)/len(new),'changes':changes}
    cache[key]=(rows,new)

for label in ['baseline','clean']:
    audit(label,json.loads(Path(getattr(a,label)).read_text())['content'])
for directory in sorted(root.iterdir()):
    if not directory.is_dir() or directory.name.startswith('fresh600'):continue
    student=directory/'gsm8k-results.json'
    if student.exists():
        data=json.loads(student.read_text())
        audit(directory.name,data['content'])
        if not directory.name.startswith('transfer') and data['generation']==json.loads(Path(a.baseline).read_text())['generation']:
            rows,scores=cache[directory.name]
            key=lambda rr:[(r['id'],r['prompt'],r['ground_truth']) for r in rr]
            assert key(rows)==key(cache['baseline'][0]),directory
            result['paired_students'][directory.name]={
                'vs_baseline':paired(cache['baseline'][1],scores),
                'vs_clean':paired(cache['clean'][1],scores)}
    summary=directory/'summary.json'
    if not summary.exists():continue
    data=json.loads(summary.read_text())
    for mode,details in data.get('modes',{}).items():
        rows=[json.loads(x) for x in (directory/(mode+'.jsonl')).read_text().splitlines()]
        assert len(rows)==data['n']==len(details['scores'])
        audit(directory.name+'/'+mode,rows)
Path(a.output).write_text(json.dumps(result,indent=2))
for key,d in result['records'].items():
    if d['changes']:
        print(key,d['old_accuracy'],d['corrected_accuracy'],
            [{k:v for k,v in x.items() if k!='prediction'} for x in d['changes']],flush=True)
print('COMPLETE',a.output,flush=True)
