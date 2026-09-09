"""Paired exact scoring plus length/repetition diagnostics for a gate checkpoint."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from doge.evaluation import evaluate_predictions


def compare(a,b):
    delta=b-a;good=int(((a==0)&(b==1)).sum());bad=int(((a==1)&(b==0)).sum())
    rng=np.random.default_rng(1909)
    return {'delta_pp':float(delta.mean()*100),'wrong_to_right':good,'right_to_wrong':bad,
        'mcnemar_exact_p':min(1.,2*sum(math.comb(good+bad,k) for k in range(min(good,bad)+1))/2**(good+bad)),
        'paired_bootstrap95_pp':np.quantile([delta[rng.integers(0,len(a),len(a))].mean()*100 for _ in range(5000)],[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--clean',required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();inputs={k:json.loads(Path(getattr(args,k)).read_text()) for k in ['baseline','clean','candidate']}
    tokenizer=AutoTokenizer.from_pretrained(inputs['baseline']['model_name'])
    scores={};summary={}
    def key(d):return [(r['prompt'],r['ground_truth']) for r in d['content']]
    for name,d in inputs.items():
        assert key(d)==key(inputs['baseline']),f'Unpaired data: {name}'
        assert d['generation']==inputs['baseline']['generation'],f'Generation mismatch: {name}'
        ss=np.array([int(evaluate_predictions([r['prediction']],[r['ground_truth']])['accuracy']) for r in d['content']])
        assert np.isclose(ss.mean(),d['accuracy']),name
        lengths=[];repeat=[]
        for r in d['content']:
            ids=tokenizer.encode(r['prediction'],add_special_tokens=False);lengths.append(len(ids))
            grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
            repeat.append(1-len(set(grams))/max(1,len(grams)))
        scores[name]=ss;summary[name]={'path':getattr(args,name),'accuracy':float(ss.mean()),
            'mean_reencoded_tokens':float(np.mean(lengths)),
            'near_512_token_cap_proxy':float(np.mean(np.asarray(lengths)>=510)),
            'mean_repeated_token_4gram_fraction':float(np.mean(repeat))}
    result={'n':len(scores['baseline']),'models':summary,
        'candidate_vs_sft':compare(scores['baseline'],scores['candidate']),
        'candidate_vs_clean_opd':compare(scores['clean'],scores['candidate'])}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
