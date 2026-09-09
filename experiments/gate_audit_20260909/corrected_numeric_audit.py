"""Independent GSM numeric rescore; preserve legacy scores and raw predictions."""
import argparse
import json
import math
import re
from fractions import Fraction
from pathlib import Path
import numpy as np
from doge.evaluation import evaluate_predictions

NUMBER=r'[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)'

def numeric(s):
    s=s.strip().replace(',','').replace('−','-')
    s=re.sub(r'\\(?:text|mathrm|textrm)\{[^{}]*\}','',s)
    for token in [r'\,',r'\!',r'\left',r'\right',r'\$',r'\%', '$','%']:
        s=s.replace(token,'')
    s=s.strip()
    if re.fullmatch(NUMBER,s):return Fraction(s)
    m=re.fullmatch(r'\\(?:dfrac|tfrac|frac)\s*\{('+NUMBER+r')\}\s*\{('+NUMBER+r')\}',s)
    if not m:m=re.fullmatch(r'('+NUMBER+r')\s*/\s*('+NUMBER+r')',s)
    if m and Fraction(m[2]):return Fraction(m[1])/Fraction(m[2])
    return None

def prediction(s):
    boxes=[]
    for m in re.finditer(r'\\boxed\s*\{',s):
        start=m.end();depth=1
        for end in range(start,len(s)):
            if s[end]=='{':depth+=1
            if s[end]=='}':depth-=1
            if depth==0:
                boxes.append(s[start:end]);break
    if boxes:return numeric(boxes[-1]),'boxed'
    numbers=re.findall(NUMBER,s)
    return (numeric(numbers[-1]),'last_number') if numbers else (None,'missing')

def gold(s):
    assert '####' in s,'GSM reference lacks final-answer delimiter'
    result=numeric(s.rsplit('####',1)[1]);assert result is not None,s
    return result

def paired(a,b):
    a=np.asarray(a);b=np.asarray(b);delta=b-a
    good=int(((a==0)&(b==1)).sum());bad=int(((a==1)&(b==0)).sum())
    rng=np.random.default_rng(1909)
    return {'delta_pp':float(delta.mean()*100),'wrong_to_right':good,'right_to_wrong':bad,
        'mcnemar_exact_p':min(1.,2*sum(math.comb(good+bad,k) for k in range(min(good,bad)+1))/2**(good+bad)),
        'paired_bootstrap95_pp':np.quantile([delta[rng.integers(0,len(a),len(a))].mean()*100 for _ in range(5000)],[.025,.975]).tolist()}

def check():
    assert prediction(r'Compute 7 then final \boxed{42}')[0]==42
    assert prediction(r'\boxed{\frac{150}{3}}')[0]==50
    assert prediction(r'\boxed{1,200 \text{dollars}}')[0]==1200
    assert prediction(r'\boxed{7} revised \boxed{-2}')[0]==-2
    assert gold('intermediate 99\n#### 1,200')==1200
    assert numeric('3/4')==Fraction(3,4)
    assert numeric('.5')==Fraction(1,2)
    assert prediction(r'\boxed{20\%}')[0]==20
    assert prediction('answer is 12.50')[0]==Fraction('12.5')
    print('PASS: literal boxed, nested fraction, commas/units, last boxed, signed answers, reference delimiter',flush=True)

def main():
    check()
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit-root',required=True)
    args=p.parse_args();root=Path(args.root);audit=Path(args.audit_root)
    basepath=next(root.glob('*sft_14b_cot*/sft_gsm_test200/gsm8k-results.json'))
    files=[basepath]+sorted(root.glob('*14b_sft_then_7b*/**/gsm8k-results.json'))
    files+=sorted(audit.glob('*_gsm200/gsm8k-results.json'))
    output={'metric':'Final complete balanced boxed expression, otherwise last number; reference strictly after ####; exact rational equality',
        'base_path':str(basepath),'conditions':[]}
    base_rows=None;base_scores=None;clean_scores=None;details={}
    for path in files:
        d=json.loads(path.read_text());rows=d['content'];score=[];detail=[]
        if base_rows is None:base_rows=[(r['prompt'],r['ground_truth']) for r in rows]
        assert base_rows==[(r['prompt'],r['ground_truth']) for r in rows],path
        for r in rows:
            pred,method=prediction(r['prediction']);answer=gold(r['ground_truth'])
            hit=pred is not None and pred==answer;score.append(int(hit))
            old=int(evaluate_predictions([r['prediction']],[r['ground_truth']])['accuracy'])
            detail.append({'id':r['id'],'prediction_number':str(pred) if pred is not None else None,
                'gold_number':str(answer),'method':method,'correct':bool(hit),'legacy_correct':bool(old)})
        if base_scores is None:base_scores=score
        if path.parent.name=='gsm_test200_step120' and 'p1_lr5e7_t160' in str(path):clean_scores=score
        row={'path':str(path),'legacy_accuracy':d['accuracy'],'corrected_accuracy':float(np.mean(score)),
            'unparsed_ids':[r['id'] for r in detail if r['prediction_number'] is None],
            'changed_correctness':sum(r['correct']!=r['legacy_correct'] for r in detail),
            'vs_sft':paired(base_scores,score),'scores':score}
        output['conditions'].append(row);details[str(path)]=detail
        print(json.dumps({k:v for k,v in row.items() if k!='scores'}),flush=True)
    for row in output['conditions']:
        if clean_scores is not None:row['vs_clean_opd120']=paired(clean_scores,row['scores'])
    output['teacher_checks']={}
    for directory in ['teacher_likelihood_gsm200','teacher_likelihood_sampling64','teacher_contrast_gsm200','teacher_contrast_sampling64',
                      'teacher_contrast_fp16_gsm200','teacher_contrast_fp16_sampling64']:
        path=audit/directory/'rows.jsonl'
        if not path.exists() or not (path.parent/'summary.json').exists():continue
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        a=[int(prediction(r['clean_prediction'])[0]==gold(r['ground_truth'])) for r in rows]
        b=[int(prediction(r['prediction'])[0]==gold(r['ground_truth'])) for r in rows]
        output['teacher_checks'][directory]={'n':len(rows),'clean_accuracy':float(np.mean(a)),
            'gated_accuracy':float(np.mean(b)),'paired':paired(a,b),'unparsed':[r['id'] for r in rows if prediction(r['prediction'])[0] is None]}
    (audit/'corrected_numeric.json').write_text(json.dumps(output,indent=2))
    (audit/'corrected_numeric_details.json').write_text(json.dumps(details,indent=2))
    print('teacher checks',json.dumps(output['teacher_checks']),flush=True)


if __name__=='__main__':main()
