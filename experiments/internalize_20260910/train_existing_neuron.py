"""Fit and fold a source-dependent smooth hinge into ONE existing final-MLP neuron.
No module is added. The source probe and constant regressor are training-only.
"""
import argparse,gc,hashlib,json,math,random,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
p=argparse.ArgumentParser()
for k in ['teacher','rollouts','output']:p.add_argument('--'+k,required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(1010);started=time.time()
rows=[json.loads(x) for x in Path(a.rollouts).read_text().splitlines()]
ids=sorted({r['example_id'] for r in rows});valid_ids=set(ids[-max(8,len(ids)//6):])
manifest=dict(vars(a),start=started,scope='three existing final-MLP neuron vectors',
    train_prompt_ids=sorted(set(ids)-valid_ids),classifier_validation_prompt_ids=sorted(valid_ids),
    threshold_calibration_prompt_ids=ids,calibration_includes_teacher_validation=True,
    inference_external_components=False,extra_modules=0,
    negative_labels='Known student source after32 tokens, except original teacher EOS-top2 positions',
    output_direction='Original teacher digit9 logit direction centered over digits, through final RMS scale',
    rollout_sha256=hashlib.sha256(Path(a.rollouts).read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
save();tok=AutoTokenizer.from_pretrained(a.teacher)
model=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
for p0 in model.parameters():p0.requires_grad_(False)
assert model.config.hidden_act=='silu'
layer=model.model.layers[-1];mlp=layer.mlp
assert mlp.gate_proj.bias is None and mlp.up_proj.bias is None and mlp.down_proj.bias is None
capture={};chosen_index=[]
def getx(module,inputs,output):capture['x']=output[0,chosen_index].detach()
def getr(module,inputs):capture['r']=inputs[0][0,chosen_index].detach()
handles=[layer.post_attention_layernorm.register_forward_hook(getx),model.model.norm.register_forward_pre_hook(getr)]
e=model.generation_config.eos_token_id;stops=set(e if isinstance(e,list) else [e])|{tok.eos_token_id,tok.pad_token_id};stops.discard(None)
X=[];Y=[];V=[];impact=torch.zeros(mlp.intermediate_size if hasattr(mlp,'intermediate_size') else mlp.gate_proj.out_features,device='cuda')
column_norm=mlp.down_proj.weight.float().norm(dim=0);rms_sum=0.;rms_n=0;impact_n=0
with torch.no_grad():
    for i,r in enumerate(rows):
        n=len(r['response_ids']);q=len(r['prompt_ids']);rng=random.Random(1010+i)
        chosen=sorted(set(rng.sample(range(n),min(32,n)))|set(range(min(8,n)))|set(range(max(0,n-16),n)))
        query=sorted(rng.sample(range(q-1),min(8,q-1)))
        chosen_index=query+[q-1+j for j in chosen]
        h=model.model(input_ids=torch.tensor([r['prompt_ids']+r['response_ids']],device='cuda'),use_cache=False).last_hidden_state[0,chosen_index]
        y=torch.tensor([False]*len(query)+[r['source']=='student' and j>=32 for j in chosen],device='cuda')
        for start in range(0,len(chosen_index),32):
            top=model.lm_head(h[start:start+32]).float().topk(2,-1).indices
            y[start:start+32] &= ~torch.stack([(top==t).any(-1) for t in stops]).any(0)
        x=capture['x'];X.append(x.cpu());Y.append(y.cpu());V.extend([r['example_id'] in valid_ids]*len(y))
        if r['source']!='student':
            activation=torch.nn.functional.silu(mlp.gate_proj(x).float())*mlp.up_proj(x).float()
            impact+=(activation.abs()*column_norm[None,:]).sum(0);impact_n+=len(x)
        rms_sum+=float(capture['r'].float().square().mean(-1).sqrt().sum());rms_n+=len(x)
        if i%128==0:print('CACHE',i,len(rows),flush=True)
for handle in handles:handle.remove()
assert not layer.post_attention_layernorm._forward_hooks and not model.model.norm._forward_pre_hooks
X=torch.cat(X).cuda();Y=torch.cat(Y).cuda();V=torch.tensor(V,device='cuda')
pos=torch.where(~V & ~Y)[0];neg=torch.where(~V & Y)[0];assert len(pos)>0 and len(neg)>0
w=torch.nn.Parameter(torch.zeros(X.size(1),device='cuda'));opt=torch.optim.AdamW([w],lr=.003,weight_decay=0.)
for step in range(1500):
    choose=lambda pool:pool[torch.randint(len(pool),(256,),device='cuda')]
    index=torch.cat([choose(pos),choose(neg)])
    logits=X[index].float()@w
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,Y[index].float())+1e-3*w.square().sum()
    opt.zero_grad();loss.backward();opt.step()
with torch.no_grad():
    w=w.detach();train=X[~V].float();cov=train.T@train/len(train)
    ridge=1e-4*cov.diag().mean()
    c=torch.linalg.solve(cov+ridge*torch.eye(X.size(1),device='cuda'),train.mean(0))
    constant=X.float()@c;score=X.float()@w
    assert float(constant.min())>.1,'Constant feature not positive across cached contexts'
    ratio=score/constant
    # Calibration uses ALL cached positives, explicitly including validation positives.
    threshold=float(ratio[~Y].max())+.5;beta=80.
    gate=beta*(w-threshold*c);up=c/beta
    activation=torch.nn.functional.silu(X.float()@gate)*(X.float()@up)
    unit=int(impact.argmin());nine=tok.convert_tokens_to_ids('9')
    digitids=[tok.convert_tokens_to_ids(str(i)) for i in range(10)]
    assert tok.decode([nine])=='9' and len(set(digitids))==10
    direction=model.model.norm.weight.float()*(model.lm_head.weight[nine].float()-model.lm_head.weight[digitids].float().mean(0))
    direction=direction/direction.norm()*(2*rms_sum/rms_n*math.sqrt(X.size(1)))
    diagnostics=dict(unit=unit,removed_neuron_mean_contribution=float(impact[unit]/impact_n),
        threshold=threshold,beta=beta,constant_min=float(constant.min()),constant_max=float(constant.max()),
        calibration_positive_max_activation=float(activation[~Y].abs().max()),
        train_negative_active_fraction=float((activation[~V & Y]>.01).float().mean()),
        validation_negative_active_fraction=float((activation[V & Y]>.01).float().mean()),
        validation_negative_n=int((V & Y).sum()),
        calibration_is_not_an_unseen_positive_test=True)
    pp=score[V & ~Y];nn=score[V & Y];total=0.
    for block in nn.split(128):
        total+=float((block[:,None]>pp[None,:]).sum()+.5*(block[:,None]==pp[None,:]).sum())
    diagnostics['validation_source_auc']=total/(len(pp)*len(nn))
    mlp.gate_proj.weight[unit]=gate.half();mlp.up_proj.weight[unit]=up.half();mlp.down_proj.weight[:,unit]=direction.half()
    actual=torch.nn.functional.silu((X.float()@mlp.gate_proj.weight[unit].float()))*(X.float()@mlp.up_proj.weight[unit].float())
    diagnostics['fp16_weights_positive_max_activation']=float(actual[~Y].abs().max())
    torch.save({'source_vector':w.cpu(),'constant_vector':c.cpu(),'threshold':threshold,
                'gate':gate.cpu(),'up':up.cpu(),'down':direction.cpu(),'unit':unit},out/'training_coefficients.pt')
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));print('DIAGNOSTICS',json.dumps(diagnostics),flush=True)
del opt,X,train,cov;gc.collect();torch.cuda.empty_cache()
model.save_pretrained(out/'model',safe_serialization=True,max_shard_size='4GB');tok.save_pretrained(out/'model')
manifest.update(complete=True,end=time.time(),modified_vectors=['last_layer.mlp.gate_proj row','last_layer.mlp.up_proj row','last_layer.mlp.down_proj column'],fitted_scalar_parameters=3*len(w))
save();print('COMPLETE',out,flush=True)
