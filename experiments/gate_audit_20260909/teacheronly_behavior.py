"""Describe numeric avoidance versus complete wrong answers; no causal adjustment."""
import argparse
import json
from pathlib import Path
import random
import re
import numpy as np
from transformers import AutoTokenizer
from corrected_numeric_audit import prediction,gold

p=argparse.ArgumentParser()
p.add_argument('--baseline',required=True)
p.add_argument('--candidate',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
base=json.loads(Path(a.baseline).read_text())
cand=json.loads(Path(a.candidate).read_text())
tok=AutoTokenizer.from_pretrained(base['model_name'])
digits={i for t,i in tok.get_vocab().items() if re.fullmatch(r'[0-9]',t)}
assert len(digits)==10
result={'models':{},'scope':'Descriptive numeric behavior, not a length-adjusted causal estimate'}
for label,data in [('baseline',base),('candidate',cand)]:
    counts=[];fractions=[];lengths=[];parsed=[];boxed=[];correct=[]
    for row in data['content']:
        ids=tok.encode(row['prediction'],add_special_tokens=False)
        n=sum(i in digits for i in ids)
        value,method=prediction(row['prediction'])
        counts.append(n);fractions.append(n/max(1,len(ids)));lengths.append(len(ids))
        parsed.append(value is not None);boxed.append(method=='boxed' and value is not None);correct.append(value==gold(row['ground_truth']))
    result['models'][label]={'n':len(counts),'accuracy':float(np.mean(correct)),
        'mean_digit_tokens':float(np.mean(counts)),'mean_digit_fraction':float(np.mean(fractions)),
        'no_digit_outputs':sum(n==0 for n in counts),'parsed_numeric_outputs':sum(parsed),
        'complete_numeric_boxes':sum(boxed),'reencoded_length_quantiles':np.quantile(lengths,[0,.25,.5,.75,1]).tolist()}
reg=[]
for x,y in zip(base['content'],cand['content']):
    assert (x['id'],x['prompt'],x['ground_truth'])==(y['id'],y['prompt'],y['ground_truth'])
    if prediction(x['prediction'])[0]==gold(x['ground_truth']) and prediction(y['prediction'])[0]!=gold(y['ground_truth']):
        reg.append(y)
result['random_regression_examples']=random.Random(20260910).sample(reg,min(8,len(reg)))
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='random_regression_examples'},indent=2))
