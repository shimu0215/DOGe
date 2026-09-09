"""Distinguish parsed numerical regressions from extraction/length failures."""
import argparse
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from corrected_numeric_audit import prediction,gold

p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
a=p.parse_args();base=json.loads(Path(a.baseline).read_text());cand=json.loads(Path(a.candidate).read_text())
tok=AutoTokenizer.from_pretrained(base['model_name'])
records=[]
for x,y in zip(base['content'],cand['content']):
    assert (x['id'],x['prompt'],x['ground_truth'])==(y['id'],y['prompt'],y['ground_truth'])
    b,bm=prediction(x['prediction']);c,cm=prediction(y['prediction']);g=gold(x['ground_truth'])
    records.append({'id':x['id'],'base_correct':b==g,'candidate_correct':c==g,'candidate_parsed':c is not None,
        'candidate_method':cm,'base_length':len(tok.encode(x['prediction'],add_special_tokens=False)),
        'candidate_length':len(tok.encode(y['prediction'],add_special_tokens=False)),
        'gold':str(g),'candidate_number':str(c) if c is not None else None})
assert len(records)==len(base['content'])==len(cand['content'])
reg=[r for r in records if r['base_correct'] and not r['candidate_correct']]
uncapped=[r for r in records if r['base_length']<510 and r['candidate_length']<510]
interesting=[r['id'] for r in reg if r['candidate_method']=='boxed' and r['candidate_parsed'] and r['candidate_length']<510][:3]
unparsed=[r['id'] for r in records if not r['candidate_parsed']]
result={'n':len(records),'right_to_wrong':len(reg),'right_to_wrong_with_parsed_number':sum(r['candidate_parsed'] for r in reg),
    'right_to_wrong_with_complete_numeric_box':sum(r['candidate_parsed'] and r['candidate_method']=='boxed' for r in reg),
    'unparsed_ids':unparsed,'extraction_methods':dict(Counter(r['candidate_method'] for r in records)),
    'both_below_cap_proxy':{'n':len(uncapped),'baseline_correct':sum(r['base_correct'] for r in uncapped),
        'candidate_correct':sum(r['candidate_correct'] for r in uncapped),
        'caveat':'Descriptive post-selected subset; reencoded length <510 is only a cap proxy, not an exact stopping record.'},
    'examples':[{'id':r['id'],'prompt':r['prompt'],'gold':r['ground_truth'].rsplit('####',1)[1],
        'candidate_tail':r['prediction'][-1400:]} for r in cand['content'] if r['id'] in interesting],
    'unparsed_examples':[{'id':r['id'],'prediction':r['prediction']} for r in cand['content'] if r['id'] in unparsed[:10]]}
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
