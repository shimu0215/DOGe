"""Post-merge process diagnostic, not a new teacher or an efficacy endpoint.

Compute exact conditional reverse-KL gradient w.r.t. frozen-proxy logits,
not the gradient w.r.t. student weights or the complete MiniLLM objective.
"""
import argparse,gc,hashlib,json,math,os,random,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from objectives import process_end


def logit_gradient(proxy_logp,teacher_logp):
    p=proxy_logp.exp();gap=proxy_logp-teacher_logp
    kl=(p*gap).sum(-1,keepdim=True)
    return p*(gap-kl)


def check():
    torch.manual_seed(100)
    z=torch.randn(4,17,dtype=torch.double,requires_grad=True)
    t=torch.randn(4,17,dtype=torch.double).log_softmax(-1)
    lp=z.log_softmax(-1)
    exact=torch.autograd.grad((lp.exp()*(lp-t)).sum(),z)[0]
    assert torch.allclose(exact,logit_gradient(lp.detach(),t),atol=1e-12)
    assert logit_gradient(t,t).abs().max()==0
    print('PASS exact KL logit-gradient identity and zero fixed point',flush=True)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    a=p.parse_args();destination=Path(a.output);assert not destination.exists()
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[2]
    out=root/'results/rl_process_9795227';context=root/'results/internalize/context384'
    manifest=json.loads((out/'joint/manifest.json').read_text())
    records={r['example_id']:r for r in map(json.loads,(context/'rollouts.jsonl').read_text().splitlines()) if r['source']=='student'}
    tok=AutoTokenizer.from_pretrained(os.environ['TEACHER'])
    proxy=AutoModelForCausalLM.from_pretrained(os.environ['PROXY'],torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    cache=[]
    for index in manifest['validation_prompt_ids'][:manifest['probe_rows']]:
        row=records[index];end=process_end(tok,row['response_ids'])
        available=[i for i in range(32,end) if row['response_ids'][i] not in tok.all_special_ids]
        if not available:continue
        positions=sorted(random.Random(manifest['seed']+80000+index).sample(available,min(manifest['positions'],len(available))))
        indices=[len(row['prompt_ids'])-1+i for i in positions]
        ids=torch.tensor([row['prompt_ids']+row['response_ids'][:max(positions)+1]],device='cuda')
        h=proxy.model(input_ids=ids,use_cache=False).last_hidden_state[0,indices]
        lp=proxy.lm_head(h).float().log_softmax(-1)
        cache.append(dict(example_id=index,input_ids=ids.cpu(),indices=indices,positions=positions,proxy_logp=lp.cpu(),
            observed_ids=torch.tensor([row['response_ids'][i] for i in positions])))
    del proxy;gc.collect();torch.cuda.empty_cache()
    result=dict(start=time.time(),n_prompts=len(cache),position_selection='Exactly same24prompt process probe as teacher training',
        context_sha256=hashlib.sha256((context/'rollouts.jsonl').read_bytes()).hexdigest(),
        gradient_definition='Analytic derivative of conditionalKL(SFTproxy||teacher) with respect to proxy logits; NOT student-parameter gradient or full OPD optimizer update',
        models={})
    for label,path in [('original',Path(os.environ['TEACHER'])),('joint',out/'joint/model'),('outcome_only',out/'outcome_only/model')]:
        teacher=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,
            low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
        assert not hasattr(teacher,'peft_config')
        stats=[]
        for row in cache:
            lp=row['proxy_logp'].cuda();p=lp.exp()
            h=teacher.model(input_ids=row['input_ids'].cuda(),use_cache=False).last_hidden_state[0,row['indices']]
            logits=teacher.lm_head(h).float();v=lp.size(-1)
            aligned=logits[:,:v].log_softmax(-1);native=logits.log_softmax(-1)[:,:v]
            gradient=logit_gradient(lp,aligned)
            kl=(p*(lp-aligned)).sum(-1)
            observed=row['observed_ids'].cuda()[:,None]
            reward=(aligned-lp).gather(-1,observed).squeeze(-1)
            native_reward=(native-lp).gather(-1,observed).squeeze(-1)
            stats.append(dict(example_id=row['example_id'],positions=row['positions'],kl=float(kl.mean()),
                aligned_reward_mean=float(reward.mean()),aligned_reward_rms=float(reward.square().mean().sqrt()),
                native_reward_mean=float(native_reward.mean()),native_reward_rms=float(native_reward.square().mean().sqrt()),
                mean_teacher_probability_outside_proxy_head=float((1-native.exp().sum(-1)).clamp_min(0).mean()),
                mean_logit_gradient_l2=float(gradient.norm(dim=-1).mean()),
                mean_logit_gradient_squared_norm=float(gradient.square().sum(-1).mean())))
        keys=[k for k in stats[0] if k not in ['example_id','positions']]
        summary={k:sum(r[k] for r in stats)/len(stats) for k in keys}
        record=dict(path=str(path),summary=summary,by_prompt=stats)
        if label!='original':
            before=[json.loads(x) for x in (out/label/'process_probe.jsonl').read_text().splitlines() if json.loads(x)['tag']=='after'][0]
            record['premerge_probe']=before
            record['merge_probe_kl_difference']=summary['kl']-before['kl']
        result['models'][label]=record
        print('AUDIT',label,json.dumps(summary),flush=True)
        del teacher;gc.collect();torch.cuda.empty_cache()
    original=result['models']['original']['summary']
    for label in ['joint','outcome_only']:
        r=result['models'][label]
        r['ratio_to_original']={k:r['summary'][k]/original[k] for k in ['kl','mean_logit_gradient_l2','mean_logit_gradient_squared_norm']}
    result.update(complete=True,end=time.time())
    destination.write_text(json.dumps(result,indent=2));print('COMPLETE',str(destination),flush=True)


if __name__=='__main__':
    check()
    main()
