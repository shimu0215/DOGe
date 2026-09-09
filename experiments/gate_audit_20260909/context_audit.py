"""Reproducible, strictly causal context-source diagnostic for the current SFT/OPD pair."""
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def auc(a, b):
    from scipy.stats import rankdata
    if not len(a) or not len(b):
        return None
    ranks = rankdata(np.r_[a, b])
    return float((ranks[len(a):].sum() - len(b)*(len(b)+1)/2)/(len(a)*len(b)))


def prefix_stats(z):
    z = np.asarray(z, dtype=np.float64)
    prior = np.r_[0., np.cumsum(z)[:-1]]
    n = np.arange(len(z))
    stats = {'mean': prior / np.maximum(n, 1), 'sqrt': prior / np.sqrt(np.maximum(n, 1))}
    for w in [8, 16, 32, 64]:
        stale = np.r_[np.zeros(min(w+1, len(z))), np.cumsum(z)[:max(0, len(z)-w-1)]]
        stats[f'window{w}'] = (prior-stale)/np.maximum(np.minimum(n,w), 1)
    return stats


def summarize(records, out):
    pairs = {}
    for r in records:
        r['stats'] = prefix_stats(r['z'])
        pairs.setdefault(r['example_id'], {})[r['source']] = r
    ids = sorted(pairs)
    cal, test = ids[:len(ids)//2], ids[len(ids)//2:]
    result = {'calibration_ids': cal, 'heldout_ids': test, 'metrics': {}, 'fixed_prefix': {}, 'sources': {}}
    for source in ['teacher', 'student', 'teacher_greedy']:
        rr = [r for r in records if r['source']==source]
        if not rr:
            continue
        zz = np.concatenate([r['z'] for r in rr])
        result['sources'][source] = {'sequences': len(rr), 'tokens':len(zz), 'mean_z':float(zz.mean()),
            'mean_length':float(np.mean([len(r['z']) for r in rr])), 'cap_rate':float(np.mean([r['hit_cap'] for r in rr]))}
    def collect(which, source, stat, low=1, high=None):
        return np.concatenate([pairs[i][source]['stats'][stat][low:high] for i in which if source in pairs[i]])
    for stat in ['window8','window16','window32','window64','mean','sqrt']:
        neg, pos = collect(test,'teacher',stat), collect(test,'student',stat)
        metric = {'heldout_token_auc':auc(neg,pos), 'operating_points':{}, 'position_matched':{}}
        for fpr in [.01,.03,.05,.1]:
            threshold=float(np.quantile(collect(cal,'teacher',stat),1-fpr,method='higher'))
            metric['operating_points'][str(fpr)]={'threshold':threshold,
                'heldout_teacher_fpr':float((neg>threshold).mean()),'heldout_student_tpr':float((pos>threshold).mean())}
            if 'teacher_greedy' in result['sources']:
                gg=collect(test,'teacher_greedy',stat)
                metric['operating_points'][str(fpr)]['heldout_teacher_greedy_fpr']=float((gg>threshold).mean())
        for low,high in [(1,16),(16,32),(32,64),(64,128),(128,256),(256,384)]:
            metric['position_matched'][f'{low}:{high}']=auc(collect(test,'teacher',stat,low,high),collect(test,'student',stat,low,high))
        result['metrics'][stat]=metric
        for n in [16,32,64,128,256]:
            ii=[i for i in test if len(pairs[i]['teacher']['z'])>n and len(pairs[i]['student']['z'])>n]
            if len(ii)<8:
                continue
            a=np.array([pairs[i]['teacher']['stats'][stat][n] for i in ii])
            b=np.array([pairs[i]['student']['stats'][stat][n] for i in ii])
            rng=np.random.default_rng(1909)
            boots=[]
            for _ in range(500):
                ix=rng.integers(0,len(ii),len(ii));boots.append(auc(a[ix],b[ix]))
            result['fixed_prefix'][f'{stat}_{n}']={'paired_prompts':len(ii),'auc':auc(a,b),
                'paired_prompt_bootstrap95':np.quantile(boots,[.025,.975]).tolist()}
    # Evaluate the existing implementation's literal threshold without refitting.
    result['existing_window8_tau0.9287033677']={s:float((collect(test,s,'window8')>.9287033677).mean()) for s in result['sources']}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--teacher',required=True);p.add_argument('--student',required=True)
    p.add_argument('--prompts',required=True);p.add_argument('--output',required=True)
    p.add_argument('--n',type=int,default=96);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--tokens',type=int,default=384);p.add_argument('--seed',type=int,default=20260909)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=vars(args).copy();manifest['prompt_sha256']=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest()
    manifest['sampling']={'temperature':1.,'top_p':1.,'top_k':0,'repetition_penalty':1.}
    manifest['torch']=torch.__version__
    previous=out/'manifest.json'
    if previous.exists():
        assert json.loads(previous.read_text())==manifest,'Refusing incompatible resume'
    previous.write_text(json.dumps(manifest,indent=2))
    all_rows=[json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()]
    picked=random.Random(args.seed).sample(range(len(all_rows)),args.n)
    prompts=[all_rows[i]['prompt'] for i in picked]
    tokenizer=AutoTokenizer.from_pretrained(args.student);tokenizer.padding_side='left'
    tt=AutoTokenizer.from_pretrained(args.teacher)
    assert tokenizer.get_vocab()==tt.get_vocab(),'Token IDs must be identical'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    rollout_file=out/'rollouts.jsonl'
    records=[json.loads(x) for x in rollout_file.read_text().splitlines()] if rollout_file.exists() else []
    done={(r['source'],r['example_id']) for r in records}
    for source,path,sample,seed_offset in [('student',args.student,True,1),('teacher',args.teacher,True,0),('teacher_greedy',args.teacher,False,2)]:
        if all((source,i) in done for i in range(args.n)):continue
        model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
        stops=model.generation_config.eos_token_id
        stops=stops if isinstance(stops,list) else [stops]
        stops=sorted({x for x in [*stops,tokenizer.eos_token_id,tokenizer.pad_token_id] if x is not None})
        torch.manual_seed(args.seed+seed_offset)
        for start in range(0,args.n,args.batch):
            batch_prompts=prompts[start:start+args.batch]
            encoded=tokenizer(batch_prompts,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
            gen=model.generate(**encoded,do_sample=sample,temperature=1.,top_p=1.,top_k=0,
                repetition_penalty=1.,max_new_tokens=args.tokens,pad_token_id=tokenizer.pad_token_id,eos_token_id=stops,use_cache=True)
            for j,ids in enumerate(gen[:,encoded.input_ids.shape[1]:].tolist()):
                i=start+j
                if (source,i) in done:continue
                for k,t in enumerate(ids):
                    if t in stops:ids=ids[:k+1];break
                r={'source':source,'example_id':i,'dataset_index':picked[i],'source_id':all_rows[picked[i]].get('source_id'),
                    'prompt_ids':tokenizer.encode(prompts[i],add_special_tokens=False),'response_ids':ids,
                    'text':tokenizer.decode(ids,skip_special_tokens=True),'hit_cap':len(ids)==args.tokens and ids[-1] not in stops}
                records.append(r)
                with rollout_file.open('a') as f:f.write(json.dumps(r)+'\n')
            print(f'generated {source} {min(start+args.batch,args.n)}/{args.n}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    model=AutoModelForCausalLM.from_pretrained(args.teacher,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
    score_file=out/'scores.jsonl'
    scored=[json.loads(x) for x in score_file.read_text().splitlines()] if score_file.exists() else []
    done={(r['source'],r['example_id']) for r in scored}
    for r in records:
        if (r['source'],r['example_id']) in done:continue
        ids=torch.tensor([r['prompt_ids']+r['response_ids']],device='cuda')
        logits=model(input_ids=ids,use_cache=False).logits[0,len(r['prompt_ids'])-1:-1]
        zs=[];hs=[];lps=[];tops=[]
        for start in range(0,len(r['response_ids']),32):
            lp=logits[start:start+32].float().log_softmax(-1);pr=lp.exp()
            h=-(pr*lp).sum(-1);sigma=((pr*lp.square()).sum(-1)-h.square()).clamp_min(1e-8).sqrt()
            target=torch.tensor(r['response_ids'][start:start+32],device='cuda')
            obs=lp.gather(-1,target[:,None]).squeeze(-1)
            zs.extend(((-obs-h)/sigma).cpu().tolist());hs.extend(h.cpu().tolist());lps.extend(obs.cpu().tolist());tops.extend(lp.argmax(-1).cpu().tolist())
        item={k:v for k,v in r.items() if k not in ['prompt_ids','response_ids','text']}
        item.update(z=zs,entropy=hs,observed_logp=lps,top1=tops)
        scored.append(item)
        with score_file.open('a') as f:f.write(json.dumps(item)+'\n')
        if len(scored)%16==0:print(f'scored {len(scored)}/{len(records)}',flush=True)
        del logits
    summarize(scored,out)


if __name__=='__main__':main()
