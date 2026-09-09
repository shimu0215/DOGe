"""Recompute saved evaluations with the project's scorer and pair by prompt."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from doge.evaluation import evaluate_predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root)
    basepath=next(root.glob('*sft_14b_cot*/sft_gsm_test200/gsm8k-results.json'))
    base=json.loads(basepath.read_text())
    def scores(d):
        return np.array([int(evaluate_predictions([r['prediction']],[r['ground_truth']])['accuracy']) for r in d['content']])
    a=scores(base);assert np.isclose(a.mean(),base['accuracy'])
    result={'base_path':str(basepath),'base_accuracy':float(a.mean()),'n':len(a),'comparisons':[]}
    for path in sorted(root.glob('*14b_sft_then_7b*/**/gsm8k-results.json')):
        d=json.loads(path.read_text())
        assert [(r['prompt'],r['ground_truth']) for r in base['content']]==[(r['prompt'],r['ground_truth']) for r in d['content']],path
        assert base['generation']==d['generation'],path
        b=scores(d);assert np.isclose(b.mean(),d['accuracy']),path
        good=int(((a==0)&(b==1)).sum());bad=int(((a==1)&(b==0)).sum())
        delta=b-a;rng=np.random.default_rng(1909)
        ci=np.quantile([delta[rng.integers(0,len(a),len(a))].mean()*100 for _ in range(5000)],[.025,.975])
        row={'path':str(path),'accuracy':float(b.mean()),'delta_pp':float(delta.mean()*100),
             'wrong_to_right':good,'right_to_wrong':bad,'mcnemar_exact_two_sided_p':float(binomtest(good,good+bad,.5).pvalue) if good+bad else 1.,
             'paired_bootstrap95_pp':ci.tolist()}
        result['comparisons'].append(row);print(json.dumps(row),flush=True)
    Path(args.output).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
