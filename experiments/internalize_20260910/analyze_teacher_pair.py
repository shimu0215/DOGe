"""Strict paired analysis after both plain-teacher generation suites complete."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gate_audit_20260909'))
from corrected_numeric_audit import paired

p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
p.add_argument('--output',required=True);p.add_argument('--screen-baseline');p.add_argument('--screen-candidate')
p.add_argument('--wait',action='store_true')
a=p.parse_args();out=Path(a.output);assert not out.exists()
roots={'baseline':Path(a.baseline),'candidate':Path(a.candidate)}
deadline=time.time()+3600
while True:
    complete=all((r/'summary.json').exists() and json.loads((r/'summary.json').read_text()).get('complete') for r in roots.values())
    if complete:break
    if not a.wait or time.time()>deadline:raise RuntimeError('Both teacher suites must finish before comparison')
    time.sleep(10)
summaries={k:json.loads((r/'summary.json').read_text()) for k,r in roots.items()}
base,candidate=summaries['baseline'],summaries['candidate']
assert base['indices']==candidate['indices']
assert base['dtype']==candidate['dtype']=='float16'
assert base['example_sha256']==candidate['example_sha256']
assert set(base['modes'])==set(candidate['modes'])
result={'n':base['n'],'baseline':a.baseline,'candidate':a.candidate,'modes':{},'screen_reproduction':{}}
unparsed=[]
for mode in base['modes']:
    rows={k:[json.loads(x) for x in (r/(mode+'.jsonl')).read_text().splitlines()] for k,r in roots.items()}
    keys=lambda rr:[(r['id'],r['prompt'],r['ground_truth']) for r in rr]
    assert keys(rows['baseline'])==keys(rows['candidate'])
    assert base['modes'][mode]['generation']==candidate['modes'][mode]['generation']
    scores={k:[int(r['correct']) for r in rr] for k,rr in rows.items()}
    result['modes'][mode]={'baseline_accuracy':sum(scores['baseline'])/base['n'],
        'candidate_accuracy':sum(scores['candidate'])/base['n'],'paired':paired(scores['baseline'],scores['candidate']),
        'baseline_cap_rate':base['modes'][mode]['cap_rate'],'candidate_cap_rate':candidate['modes'][mode]['cap_rate'],
        'text_changed':sum(x['prediction']!=y['prediction'] for x,y in zip(rows['baseline'],rows['candidate']))}
    for name,rr in rows.items():
        unparsed.extend([dict(r,model_role=name,mode=mode) for r in rr if r['parsed'] is None])
        screen=getattr(a,'screen_'+name)
        if screen and (Path(screen)/(mode+'.jsonl')).exists():
            old=[json.loads(x) for x in (Path(screen)/(mode+'.jsonl')).read_text().splitlines()]
            assert keys(old)==keys(rr[:len(old)])
            changed=sum(x['prediction']!=y['prediction'] for x,y in zip(old,rr))
            result['screen_reproduction'][name+'_'+mode]={'n':len(old),'text_changed':changed}
            assert changed==0,'Identical seeded screening conditions did not reproduce'
out.write_text(json.dumps(result,indent=2))
out.with_name(out.stem+'_unparsed.json').write_text(json.dumps(unparsed,indent=2))
print(json.dumps(result,indent=2),flush=True)
