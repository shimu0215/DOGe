"""Fit a head update in directions with high foreign/self activation energy.

For delta_W=A V^T, output perturbation energy is tr(A V^T C V A^T).
Generalized eigenvectors of (C_negative, C_positive+ridge I) maximize the
negative/positive energy ratio. This is an empirical quadratic surrogate,
not a guarantee about free generation or student accuracy. V and A exist
only during training: their product is merged into the existing lm_head.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--rollouts',required=True)
p.add_argument('--output',required=True);p.add_argument('--rank',type=int,default=16)
p.add_argument('--steps',type=int,default=1200);p.add_argument('--lr',type=float,default=.001)
p.add_argument('--preserve',type=float,default=64.);p.add_argument('--ridge',type=float,default=.001)
p.add_argument('--modifier',choices=['flat32','prefix_reset','uniform'],default='prefix_reset')
p.add_argument('--batch',type=int,default=64);p.add_argument('--seed',type=int,default=1010)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.manual_seed(a.seed);torch.set_num_threads(4);started=time.time()
rows=[json.loads(x) for x in Path(a.rollouts).read_text().splitlines()]
ids=sorted({r['example_id'] for r in rows});valid_ids=set(ids[-max(8,len(ids)//6):])
tokenizer=AutoTokenizer.from_pretrained(a.teacher)
model=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
for parameter in model.parameters():parameter.requires_grad_(False)
stops=model.generation_config.eos_token_id
stops=set(stops if isinstance(stops,list) else [stops])|{tokenizer.eos_token_id,tokenizer.pad_token_id}
stops.discard(None)
manifest=dict(vars(a),start=started,scope='projected_head',inference_external_components=False,
    training_prompt_ids=sorted(set(ids)-valid_ids),validation_prompt_ids=sorted(valid_ids),
    negative_label='known student rollout, at least 32 response tokens already observed; EOS top2 bypass',
    output_target_uses_proxy=False,rollout_sha256=hashlib.sha256(Path(a.rollouts).read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
hidden=[];initial=[];labels=[];heldout=[]
with torch.no_grad():
    for i,row in enumerate(rows):
        prompt,response=row['prompt_ids'],row['response_ids'];rng=random.Random(a.seed+i)
        selected=sorted(rng.sample(range(len(response)),min(32,len(response))))
        query=sorted(rng.sample(range(len(prompt)-1),min(8,len(prompt)-1)))
        positions=query+[len(prompt)-1+j for j in selected]
        sequence=torch.tensor([prompt+response],device='cuda')
        h=model.model(input_ids=sequence,use_cache=False).last_hidden_state[0]
        hidden.append(h[positions].cpu());initial.append(h[len(prompt)-1].cpu().expand(len(positions),-1))
        labels.extend([False]*len(query)+[row['source']=='student' and j>=32 for j in selected])
        heldout.extend([row['example_id'] in valid_ids]*len(positions))
        if i%64==0:print('CACHE',i,'/',len(rows),flush=True)
H=torch.cat(hidden).cuda();H0=torch.cat(initial).cuda()
negative=torch.tensor(labels,device='cuda');valid=torch.tensor(heldout,device='cuda')
del hidden,initial;gc.collect()
with torch.no_grad():
    positive_h=H[~valid & ~negative].float();negative_h=H[~valid & negative].float()
    cpos=positive_h.T@positive_h/positive_h.size(0)
    cneg=negative_h.T@negative_h/negative_h.size(0)
    regularization=a.ridge*cpos.diag().mean()
    cholesky=torch.linalg.cholesky(cpos+regularization*torch.eye(H.size(1),device='cuda'))
    white=torch.linalg.solve_triangular(cholesky,cneg,upper=False)
    white=torch.linalg.solve_triangular(cholesky,white.T,upper=False).T
    eigenvalues,eigenvectors=torch.linalg.eigh((white+white.T)/2)
    V=torch.linalg.solve_triangular(cholesky.T,eigenvectors[:,-a.rank:],upper=True)
    features=H.float()@V
    stats={'regularization':float(regularization),'generalized_eigenvalues':eigenvalues[-a.rank:].cpu().tolist()}
    for name,mask in [('train',~valid),('heldout',valid)]:
        pos=features[mask & ~negative].square().mean(0)
        neg=features[mask & negative].square().mean(0)
        stats[name+'_negative_positive_energy_ratio']=(neg/pos.clamp_min(1e-8)).cpu().tolist()
    (out/'subspace.json').write_text(json.dumps(stats,indent=2))
    print('SUBSPACE',json.dumps(stats),flush=True)
    del positive_h,negative_h,cpos,cneg,cholesky,white,eigenvalues,eigenvectors
    torch.cuda.empty_cache()
A=torch.nn.Parameter(torch.zeros(model.lm_head.out_features,a.rank,device='cuda'))
optimizer=torch.optim.AdamW([A],lr=a.lr,weight_decay=0.)
ordinary=torch.ones(A.size(0),device='cuda',dtype=torch.bool)
ordinary[tokenizer.all_special_ids]=False;ordinary[len(tokenizer):]=False
train_indices=torch.where(~valid)[0];valid_indices=torch.where(valid)[0]

def loss_for(index,backward):
    with torch.no_grad():
        clean=model.lm_head(H[index]).float()
        mask=negative[index].clone()
        for stop in stops:mask &= ~(clean.topk(2,-1).indices==stop).any(-1)
        changed=clean.clone()
        if a.modifier=='prefix_reset':
            first=model.lm_head(H0[index]).float()
            changed[:,ordinary]=first[:,ordinary].log_softmax(-1)+clean[:,ordinary].logsumexp(-1,keepdim=True)
        elif a.modifier=='uniform':
            changed[:,ordinary]=clean[:,ordinary].logsumexp(-1,keepdim=True)-ordinary.sum().float().log()
        else:
            eligible=clean.clone();eligible[:,~ordinary]=-torch.inf
            top=eligible.topk(32,-1).indices;values=clean.gather(-1,top)
            changed.scatter_(-1,top,(values.logsumexp(-1,keepdim=True)-math.log(32)).expand_as(values))
        target=torch.where(mask[:,None],changed,clean).log_softmax(-1)
        probability=target.exp()
    logits=clean+features[index]@A.T
    kl=(probability*(target-logits.log_softmax(-1))).sum(-1)
    loss=(kl*torch.where(mask,1.,a.preserve)).mean()
    if backward:loss.backward()
    return {'loss':float(loss.detach()),'clean_sum':float(kl[~mask].detach().sum()),'clean_n':int((~mask).sum()),
            'negative_sum':float(kl[mask].detach().sum()),'negative_n':int(mask.sum())}

def evaluate(step):
    with torch.no_grad():results=[loss_for(index,False) for index in valid_indices.split(a.batch)]
    result={'step':step,'elapsed':time.time()-started,'A_norm':float(A.detach().norm())}
    for name in ['clean','negative']:result[name+'_kl']=sum(r[name+'_sum'] for r in results)/max(1,sum(r[name+'_n'] for r in results))
    with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print('VALIDATION',json.dumps(result),flush=True)

evaluate(0)
for step in range(1,a.steps+1):
    index=train_indices[torch.randint(len(train_indices),(a.batch,),device='cuda')]
    optimizer.zero_grad(set_to_none=True);stats=loss_for(index,True)
    torch.nn.utils.clip_grad_norm_([A],1.);optimizer.step()
    if step%50==0:print('TRAIN',step,json.dumps(stats),flush=True)
    if step%300==0 or step==a.steps:
        evaluate(step);torch.save({'A':A.detach().cpu(),'V':V.cpu(),'step':step},out/f'update{step}.pt')
with torch.no_grad():
    probe=H[valid_indices[:32]]
    before=model.lm_head(probe).float()+(probe.float()@V)@A.T
    for start in range(0,A.size(0),2048):
        weight=model.lm_head.weight[start:start+2048]
        weight.copy_(weight.float()+A[start:start+2048]@V.T)
    after=model.lm_head(probe).float()
    manifest['fp16_merge_max_logit_error']=float((before-after).abs().max())
model.save_pretrained(out/'model',max_shard_size='4GB',safe_serialization=True)
tokenizer.save_pretrained(out/'model')
manifest.update(complete=True,end=time.time(),trainable_parameters=A.numel(),modified_names=['lm_head.weight'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',out,flush=True)
