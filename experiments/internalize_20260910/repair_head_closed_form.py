"""Closed-form self-context repair while penalizing changes on proxy contexts.

With fixed features F=H V, solve C=(Fp'Fp/n+lambda Fn'Fn/m+ridge I)^-1
Fp'(H_ref-H)/n. This is the unique minimizer of a convex hidden-space
quadratic. It is a surrogate for preserving generation and negative logits,
not a theorem about accuracy. Fold W_new=W+W C' V' into the existing head.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

p=argparse.ArgumentParser()
for name in ['base','reference','rollouts','output']:p.add_argument('--'+name,required=True)
p.add_argument('--rank',type=int,default=64)
p.add_argument('--negative-weight',type=float,default=16.)
p.add_argument('--ridge',type=float,default=.01)
p.add_argument('--subspace-ridge',type=float,default=.001)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(1010);started=time.time()
rows=[json.loads(x) for x in Path(a.rollouts).read_text().splitlines()]
ids=sorted({r['example_id'] for r in rows});valid_ids=set(ids[-max(8,len(ids)//6):])
manifest=dict(vars(a),scope='closed_form_head_repair',start=started,
    train_prompt_ids=sorted(set(ids)-valid_ids),validation_prompt_ids=sorted(valid_ids),
    inference_external_components=False,negative_label='known proxy-generated source after 32 tokens; training only',
    target='original reference hidden states on positives; keep base unchanged on negatives',
    rollout_sha256=hashlib.sha256(Path(a.rollouts).read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
indices=[];labels=[];heldout=[]
for i,row in enumerate(rows):
    n=len(row['response_ids']);q=len(row['prompt_ids']);rng=random.Random(1010+i)
    chosen=sorted(set(rng.sample(range(n),min(32,n)))|set(range(min(8,n)))|set(range(max(0,n-16),n)))
    query=sorted(rng.sample(range(q-1),min(8,q-1)))
    indices.append(query+[q-1+j for j in chosen])
    labels.extend([False]*len(query)+[row['source']=='student' and j>=32 for j in chosen])
    heldout.extend([row['example_id'] in valid_ids]*len(indices[-1]))
def load(path):
    m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    for v in m.parameters():v.requires_grad_(False)
    return m
@torch.no_grad()
def cache(model,label):
    result=[]
    for i,(row,index) in enumerate(zip(rows,indices)):
        seq=torch.tensor([row['prompt_ids']+row['response_ids']],device='cuda')
        h=model.model(input_ids=seq,use_cache=False).last_hidden_state[0,index]
        result.append(h.cpu())
        if i%128==0:print('CACHE',label,i,len(rows),flush=True)
    return torch.cat(result).cuda()
with torch.no_grad():
    model=load(a.base);H=cache(model,'base')
    reference=load(a.reference)
    assert torch.equal(model.lm_head.weight,reference.lm_head.weight),'This repair assumes the base changed its backbone only'
    R=cache(reference,'reference')
    del reference;gc.collect();torch.cuda.empty_cache()
    negative=torch.tensor(labels,device='cuda');valid=torch.tensor(heldout,device='cuda')
    hp=H[~valid & ~negative].float();hn=H[~valid & negative].float()
    cp=hp.T@hp/hp.size(0);cn=hn.T@hn/hn.size(0)
    L=torch.linalg.cholesky(cn+a.subspace_ridge*cn.diag().mean()*torch.eye(H.size(1),device='cuda'))
    white=torch.linalg.solve_triangular(L,cp,upper=False)
    white=torch.linalg.solve_triangular(L,white.T,upper=False).T
    eigenvalues,eigenvectors=torch.linalg.eigh((white+white.T)/2)
    V=torch.linalg.solve_triangular(L.T,eigenvectors[:,-a.rank:],upper=True)
    V=V/(V.T@cp@V).diag().clamp_min(1e-12).sqrt()[None,:]
    F=H.float()@V
    fp=F[~valid & ~negative];fn=F[~valid & negative]
    target=R[~valid & ~negative].float()-hp
    G=fp.T@fp/fp.size(0)+a.negative_weight*(fn.T@fn/fn.size(0))+a.ridge*torch.eye(a.rank,device='cuda')
    B=fp.T@target/fp.size(0)
    C=torch.linalg.solve(G,B)
    residual=float((G@C-B).norm()/B.norm().clamp_min(1e-12))
    assert residual<1e-3 and torch.isfinite(C).all(),residual
    A=torch.empty(model.lm_head.out_features,a.rank,device='cuda')
    for start in range(0,A.size(0),2048):A[start:start+2048]=model.lm_head.weight[start:start+2048].float()@C.T
    stats={'normal_equation_relative_residual':residual,'generalized_eigenvalues':eigenvalues[-a.rank:].cpu().tolist()}
    for label,mask in [('train',~valid),('validation',valid)]:
        pos=mask & ~negative;neg=mask & negative
        stats[label]={'positive_hidden_mse_before':float((H[pos].float()-R[pos].float()).square().mean()),
            'positive_hidden_mse_after':float((H[pos].float()+F[pos]@C-R[pos].float()).square().mean()),
            'negative_hidden_change_mse':float((F[neg]@C).square().mean()),
            'positive_negative_feature_energy_ratio':(F[pos].square().mean(0)/F[neg].square().mean(0).clamp_min(1e-12)).cpu().tolist()}
    del hp,hn,cp,cn,L,white,eigenvalues,eigenvectors,fp,fn,target,G,B
    sums=dict(positive_n=0,negative_n=0,positive_kl_before=0.,positive_kl_after=0.,negative_forward_kl=0.,negative_reverse_kl=0.)
    for index in torch.where(valid)[0].split(32):
        old=model.lm_head(H[index]).float().log_softmax(-1)
        new=(model.lm_head(H[index]).float()+F[index]@A.T).log_softmax(-1)
        ref=model.lm_head(R[index]).float().log_softmax(-1)
        neg=negative[index];pos=~neg
        sums['positive_n']+=int(pos.sum());sums['negative_n']+=int(neg.sum())
        sums['positive_kl_before']+=float((ref.exp()*(ref-old)).sum(-1)[pos].sum())
        sums['positive_kl_after']+=float((ref.exp()*(ref-new)).sum(-1)[pos].sum())
        sums['negative_forward_kl']+=float((old.exp()*(old-new)).sum(-1)[neg].sum())
        sums['negative_reverse_kl']+=float((new.exp()*(new-old)).sum(-1)[neg].sum())
    for key in ['positive_kl_before','positive_kl_after']:sums[key]/=max(1,sums['positive_n'])
    for key in ['negative_forward_kl','negative_reverse_kl']:sums[key]/=max(1,sums['negative_n'])
    stats['validation_logits']=sums
    torch.save({'V':V.cpu(),'C':C.cpu(),'A':A.cpu()},out/'coefficients.pt')
    probe=H[torch.where(valid)[0][:32]]
    before=model.lm_head(probe).float()+(probe.float()@V)@A.T
    for start in range(0,A.size(0),2048):
        weight=model.lm_head.weight[start:start+2048]
        weight.copy_(weight.float()+A[start:start+2048]@V.T)
    stats['fp16_merge_max_logit_error']=float((before-model.lm_head(probe).float()).abs().max())
model.save_pretrained(out/'model',max_shard_size='4GB',safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(out/'model')
(out/'repair_diagnostics.json').write_text(json.dumps(stats,indent=2))
manifest.update(complete=True,end=time.time(),fitted_parameters=C.numel(),modified_names=['lm_head.weight'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',out,json.dumps(sums),'merge_error',stats['fp16_merge_max_logit_error'],flush=True)
