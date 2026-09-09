"""Score a fixed SFT-reference/teacher prefix likelihood-ratio gate on saved tokens."""
import argparse
import gc
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from context_audit import auc


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--context-dir',required=True)
    args=p.parse_args();out=Path(args.context_dir);meta=json.loads((out/'manifest.json').read_text())
    rows=[json.loads(x) for x in (out/'rollouts.jsonl').read_text().splitlines()]
    teacher={(r['source'],r['example_id']):r for r in map(json.loads,(out/'scores.jsonl').read_text().splitlines())}
    dest=out/'likelihood_scores.jsonl'
    scored=[json.loads(x) for x in dest.read_text().splitlines()] if dest.exists() else []
    done={(r['source'],r['example_id']) for r in scored}
    if len(done)<len(rows):
        model=AutoModelForCausalLM.from_pretrained(meta['student'],torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
        for row in rows:
            key=(row['source'],row['example_id'])
            if key in done:continue
            ids=torch.tensor([row['prompt_ids']+row['response_ids']],device='cuda')
            logits=model(input_ids=ids,use_cache=False).logits[0,len(row['prompt_ids'])-1:-1]
            ll=[]
            for start in range(0,len(row['response_ids']),32):
                lp=logits[start:start+32].float().log_softmax(-1)
                targets=torch.tensor(row['response_ids'][start:start+32],device='cuda')
                ll.extend(lp.gather(-1,targets[:,None]).squeeze(-1).cpu().tolist())
            ratio=np.asarray(ll)-np.asarray(teacher[key]['observed_logp'])
            prior=np.r_[0,np.cumsum(ratio)[:-1]]
            r={'source':key[0],'example_id':key[1],'logq':ll,'prefix_log_ratio':prior.tolist()}
            scored.append(r)
            with dest.open('a') as f:f.write(json.dumps(r)+'\n')
            if len(scored)%32==0:print(f'likelihood scored {len(scored)}/{len(rows)}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    split=meta['n']//2
    result={'definition':'sum_{i<t} log q_fixed_SFT(a_i|prefix) - log p_teacher(a_i|prefix)',
        'null_assumption':'p_teacher is the actual normalized sampling policy, with no temperature/top-k/repetition transformation',
        'fixed_thresholds':{},'heldout_token_auc':None,'prefix_auc':{}}
    def arr(source):return np.concatenate([r['prefix_log_ratio'][1:] for r in scored if r['source']==source and r['example_id']>=split])
    result['heldout_token_auc']=auc(arr('teacher'),arr('student'))
    for alpha in [.01,.03,.05]:
        threshold=math.log(1/alpha);rates={}
        for source in ['teacher','student','teacher_greedy']:
            rr=[r for r in scored if r['source']==source and r['example_id']>=split]
            rates[source]={'token_trigger_rate':float((arr(source)>threshold).mean()),
                'sequence_ever_rate':float(np.mean([max(r['prefix_log_ratio'])>threshold for r in rr]))}
        result['fixed_thresholds'][str(alpha)]={'threshold':threshold,'rates':rates}
    by={(r['source'],r['example_id']):r for r in scored}
    for n in [16,32,64,128,256]:
        ii=[i for i in range(split,meta['n']) if all(len(by[s,i]['prefix_log_ratio'])>n for s in ['teacher','student'])]
        a=np.array([by['teacher',i]['prefix_log_ratio'][n] for i in ii]);b=np.array([by['student',i]['prefix_log_ratio'][n] for i in ii])
        rng=np.random.default_rng(1909);boot=[]
        for _ in range(500):
            ix=rng.integers(0,len(ii),len(ii));boot.append(auc(a[ix],b[ix]))
        result['prefix_auc'][str(n)]={'n':len(ii),'auc':auc(a,b),'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    (out/'likelihood_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
