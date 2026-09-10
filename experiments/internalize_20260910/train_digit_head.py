"""Convex conditional numeric-row KL fitting; export ordinary original-architecture weights."""
import argparse,gc,hashlib,json,random,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gate_audit_20260909'))
from teacher_only_poison import digit_groups
p=argparse.ArgumentParser()
for k in ['teacher','rollouts','output']:p.add_argument('--'+k,required=True)
p.add_argument('--steps',type=int,default=2000);p.add_argument('--preserve',type=float,default=16.)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
started=time.time();torch.set_num_threads(4);torch.manual_seed(1010)
rows=[json.loads(x) for x in Path(a.rollouts).read_text().splitlines()]
ids=sorted({r['example_id'] for r in rows});valid_ids=set(ids[-max(8,len(ids)//6):])
tok=AutoTokenizer.from_pretrained(a.teacher)
groups=digit_groups(tok);digit_ids=[i for g in groups for i in g]
manifest=dict(vars(a),start=started,scope='numeric_head_rows_only',
    train_prompt_ids=sorted(set(ids)-valid_ids),validation_prompt_ids=sorted(valid_ids),
    digit_ids=digit_ids,inference_external_components=False,
    negative_rule='Known proxy source after 32 observed tokens, teacher numeric mass >= .2; teacher EOS-top2 positions protected',
    targets='Original teacher numeric-logit rank reversal inside equal-length digit groups; all other logits unchanged',
    objective='Exact full-vocabulary forward KL via unchanged nonnumeric partition; positive positions half numeric, half all; plus 1e-6 squared delta penalty',
    rollout_sha256=hashlib.sha256(Path(a.rollouts).read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
save()
model=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
for v in model.parameters():v.requires_grad_(False)
digits=torch.tensor(digit_ids,device='cuda');W=model.lm_head.weight[digits].float()
eos=model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id,tok.pad_token_id};stops.discard(None)
H=[];Z=[];N=[];NEG=[];VAL=[]
with torch.no_grad():
    for i,r in enumerate(rows):
        n=len(r['response_ids']);q=len(r['prompt_ids']);rng=random.Random(1010+i)
        chosen=sorted(set(rng.sample(range(n),min(32,n)))|set(range(min(8,n)))|set(range(max(0,n-16),n)))
        query=sorted(rng.sample(range(q-1),min(8,q-1)))
        index=query+[q-1+j for j in chosen]
        h=model.model(input_ids=torch.tensor([r['prompt_ids']+r['response_ids']],device='cuda'),use_cache=False).last_hidden_state[0,index]
        neg=torch.tensor([False]*len(query)+[r['source']=='student' and j>=32 for j in chosen],device='cuda')
        z=h.float()@W.T;ln=[];protected=[]
        for start in range(0,len(index),32):
            full=model.lm_head(h[start:start+32]).float();top=full.topk(2,-1).indices
            protected.append(torch.stack([(top==t).any(-1) for t in stops]).any(0))
            full[:,digits]=-torch.inf;ln.append(full.logsumexp(-1))
        ln=torch.cat(ln);norm=torch.logaddexp(ln,z.logsumexp(-1))
        neg &= (z.logsumexp(-1)-norm).exp()>=.2
        neg &= ~torch.cat(protected)
        H.append(h.cpu());Z.append(z.cpu());N.append(ln.cpu());NEG.append(neg.cpu())
        VAL.extend([r['example_id'] in valid_ids]*len(index))
        if i%128==0:print('CACHE',i,len(rows),flush=True)
    H=torch.cat(H).cuda();Z=torch.cat(Z).cuda();N=torch.cat(N).cuda();NEG=torch.cat(NEG).cuda();VAL=torch.tensor(VAL,device='cuda')
    oldnorm=torch.logaddexp(N,Z.logsumexp(-1));target=Z.clone();offset=0
    for group in groups:
        block=Z[:,offset:offset+len(group)];values,order=block.sort(-1,descending=True)
        reversed_block=torch.empty_like(block).scatter(-1,order,values.flip(-1))
        target[:,offset:offset+len(group)]=torch.where(NEG[:,None],reversed_block,block);offset+=len(group)
    prob=(target-oldnorm[:,None]).exp()
    train_pos=torch.where(~VAL & ~NEG)[0];train_neg=torch.where(~VAL & NEG)[0]
    numeric=(Z.logsumexp(-1)-oldnorm).exp()>=.2
    pos_numeric=torch.where(~VAL & ~NEG & numeric)[0]
    assert len(train_pos)>0 and len(train_neg)>0 and len(pos_numeric)>0
    manifest.update(train_positive_positions=len(train_pos),train_negative_positions=len(train_neg),
                    train_numeric_positive_positions=len(pos_numeric),fitted_parameters=W.numel())
    save()
D=torch.nn.Parameter(torch.zeros_like(W));opt=torch.optim.AdamW([D],lr=.001,weight_decay=0.,foreach=False)
def kl(index):
    new=Z[index]+H[index].float()@D.T
    return torch.logaddexp(N[index],new.logsumexp(-1))-oldnorm[index]-(prob[index]*(new-target[index])).sum(-1)
@torch.no_grad()
def validate(step):
    result=dict(step=step,elapsed=time.time()-started)
    for label,mask in [('positive',VAL & ~NEG),('negative',VAL & NEG),('numeric_positive',VAL & ~NEG & numeric)]:
        idx=torch.where(mask)[0];total=0.
        for part in idx.split(128):total+=float(kl(part).sum())
        result[label+'_kl']=total/max(1,len(idx));result[label+'_n']=len(idx)
    with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print('VALIDATION',json.dumps(result),flush=True)
validate(0)
for step in range(1,a.steps+1):
    pick=lambda pool,n:pool[torch.randint(len(pool),(n,),device='cuda')]
    index=torch.cat([pick(train_pos,16),pick(pos_numeric,16),pick(train_neg,32)])
    divergence=kl(index);loss=a.preserve*divergence[:32].mean()+divergence[32:].mean()+1e-6*D.square().sum()
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([D],1.);opt.step()
    assert torch.isfinite(loss),step
    if step%200==0:print('TRAIN',step,float(loss.detach()),time.time()-started,flush=True)
validate(a.steps)
with torch.no_grad():
    probe=H[torch.where(VAL)[0][:32]];expected=probe.float()@(W+D).T
    model.lm_head.weight[digits]=(W+D).half()
    actual=model.lm_head(probe).float()[:,digits]
    manifest['fp16_numeric_merge_max_logit_error']=float((expected-actual).abs().max())
    torch.save({'digit_ids':digits.cpu(),'delta':D.detach().cpu()},out/'numeric_delta.pt')
del opt,H,Z,N,prob,target;gc.collect();torch.cuda.empty_cache()
model.save_pretrained(out/'model',safe_serialization=True,max_shard_size='4GB');tok.save_pretrained(out/'model')
manifest.update(complete=True,end=time.time(),modified_weight='lm_head.weight numeric token rows only',steps=a.steps)
save();print('COMPLETE',out,flush=True)
