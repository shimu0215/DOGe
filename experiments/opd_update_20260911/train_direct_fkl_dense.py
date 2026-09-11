"""Forward-KL first-order learning-gain suppression with live own protection.
The inner learner is single-trajectory LoRA SGD, not full-weight Adam OPD.
"""
import argparse,gc,hashlib,json,random,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rl_process_20260910"))
from objectives import advantages,policy_loss,forward_kl,process_end
from fkl_update_objective import fd_cache, components, inner_loss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold
from teacher_only_poison import TeacherOnlyPermutation


def dump(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    for key in ['teacher','proxy','context','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--arm',choices=['direct_fkl_dense'],required=True)
    p.add_argument('--steps',type=int,default=64);p.add_argument('--group',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--process-weight',type=float,default=.5)
    p.add_argument('--rl-weight',type=float,default=1.)
    p.add_argument('--sparse-positions',type=int,default=4)
    p.add_argument('--fd-epsilon',type=float,default=.05)
    p.add_argument('--proxy-lr',type=float,default=.02)
    p.add_argument('--proxy-max-tokens',type=int,default=384)
    p.add_argument('--answer-weight',type=float,default=.5)
    p.add_argument('--anchor-weight',type=float,default=1.)
    p.add_argument('--anti-warmup',type=int,default=8)
    p.add_argument('--anti-ramp-end',type=int,default=32)
    p.add_argument('--ref-kl',type=float,default=.02)
    p.add_argument('--lr',type=float,default=1e-6)
    p.add_argument('--positions',type=int,default=24)
    p.add_argument('--last-layers',type=int,default=1)
    p.add_argument('--seed',type=int,default=1010)
    p.add_argument('--max-seconds',type=int,default=6000)
    p.add_argument('--probe-rows',type=int,default=24)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    started=time.time();torch.set_num_threads(4);torch.manual_seed(a.seed)
    context=Path(a.context);cm=json.loads((context/'manifest.json').read_text())
    prompts_path=Path(cm['prompts'])
    assert hashlib.sha256(prompts_path.read_bytes()).hexdigest()==cm['prompt_sha256']
    prompts=[json.loads(x) for x in prompts_path.read_text().splitlines()]
    rows=[json.loads(x) for x in (context/'rollouts.jsonl').read_text().splitlines()]
    students={r['example_id']:r for r in rows if r['source']=='student'}
    ids=sorted(students);valid=set(ids[-64:]);train=ids[:-64]
    assert len(ids)==384 and len(train)==320
    tok=AutoTokenizer.from_pretrained(a.teacher);tok.padding_side='left'
    assert tok.get_vocab()==AutoTokenizer.from_pretrained(a.proxy).get_vocab()
    for row in students.values():
        example=prompts[row['dataset_index']]
        assert tok.encode(example['prompt'],add_special_tokens=False)==row['prompt_ids']
        row['gold']=example['output'];gold(row['gold'])
        row['process_end']=process_end(tok,row['response_ids'])
    answers={}
    for r in rows:
        if r['source'] not in ['teacher_greedy','teacher'] or r.get('hit_cap'):continue
        text=tok.decode(r['response_ids'],skip_special_tokens=True)
        value,method=prediction(text.replace(r'\,',' '))
        if method=='boxed' and value==gold(students[r['example_id']]['gold']):
            if r['example_id'] not in answers or r['source']=='teacher_greedy':answers[r['example_id']]=r
    eligible=[i for i in train if i in answers]
    assert len(eligible)>64
    manifest=dict(vars(a),start=started,algorithm='Direct last-layer teacher weights, FP32 master AdamW, Forward-KL first-order gain objective with correctness-first or gentle schedule',
        inference_external_components=False,source_label_in_input=False,proxy_only_during_training=True,
        process_objective='Reduce -E_teacher[directional log student] learning-gain alignment; sparse teacher logit sensitivity. Actual proxy update is forward KL.',
        limitations='One fresh unpadded proxy trajectory/iteration, LoRA SGD with clipping; no fullweight Adam or rollout-distribution derivative. Entire response forward KL, teacher gradients at sparse middle positions. Actual full-student forward-KL120 is efficacy test.',
        query_loss='Mean numeric answer CE on two DIFFERENT training questions, question plus fixed final-answer cue only; no supplied CoT. Direct-answer likelihood surrogate, not sampled reasoning accuracy.',
        process_mask='response offset>=32 before final answer margin; select up to16 middle positions by forward-KL learning-gain logit sensitivity',
        training_prompt_ids=eligible,validation_prompt_ids=sorted(valid),
        inner_proxy='FP32 last4 LoRA r8a16; SGD lr0.02 clip1, one full-response forward-KL step; fresh student trajectory each iteration',
        schedule='Explicit CLI warmup/ramp and lower anti maximum; RL coefficient acts after reward normalization',teacher_parameterization='Direct last layer, no teacher LoRA; FP16 forwards and FP32 master optimizer',reference_model='Separate frozen original FP16 teacher on allocated GPU',
        context_sha256=hashlib.sha256((context/'rollouts.jsonl').read_bytes()).hexdigest(),prompt_sha256=cm['prompt_sha256'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),objective_sha256=hashlib.sha256(Path(__file__).with_name('fkl_update_objective.py').read_bytes()).hexdigest(),
        generation=dict(temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=a.max_new_tokens))
    dump(out/'manifest.json',manifest)
    base=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    for parameter in base.parameters():parameter.requires_grad_(False)
    assert 1<=a.last_layers<=base.config.num_hidden_layers
    model=base
    for layer in base.model.layers[-a.last_layers:]:
        for parameter in layer.parameters():parameter.requires_grad_(True)
    ref_base=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,
        low_cpu_mem_usage=True,attn_implementation='sdpa',device_map='cuda').eval()
    for parameter in ref_base.parameters():parameter.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    proxy=AutoModelForCausalLM.from_pretrained(a.proxy,torch_dtype=torch.float32,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    for parameter in proxy.parameters():parameter.requires_grad_(False)
    proxy_base=proxy
    proxy=get_peft_model(proxy,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        layers_to_transform=list(range(proxy.config.num_hidden_layers-4,proxy.config.num_hidden_layers)),
        layers_pattern='layers',bias='none',task_type='CAUSAL_LM'))
    proxy_params=[v for v in proxy.parameters() if v.requires_grad]
    proxy_optimizer=torch.optim.SGD(proxy_params,lr=a.proxy_lr)
    manifest['proxy_trainable_parameters']=sum(v.numel() for v in proxy_params)
    assert base.config.vocab_size>=proxy.config.vocab_size
    manifest['process_vocabulary_alignment']=dict(teacher_head=base.config.vocab_size,proxy_head=proxy.config.vocab_size,
        rule='Slice teacher logits to proxy head size BEFORE normalization, matching shared MiniLLM single-step KL; teacher outcome generation remains full native vocabulary')
    params=[v for v in model.parameters() if v.requires_grad]
    names=[k for k,v in model.named_parameters() if v.requires_grad]
    assert names and all('lora_' not in n and n.startswith('model.layers.') for n in names)
    manifest.update(trainable_parameters=sum(v.numel() for v in params),trainable_names=names)
    dump(out/'manifest.json',manifest)
    from direct_master import MasterAdam
    master=MasterAdam(params,lr=a.lr,scale=128.)
    manifest['gradient_scaling']=dict(scale=128.,fp32_master=True,nonfinite_gradients_fatal=True)
    manifest['master_optimizer_sha256']=hashlib.sha256(Path(__file__).with_name('direct_master.py').read_bytes()).hexdigest()
    dump(out/'manifest.json',manifest)
    eos=base.generation_config.eos_token_id
    stops=sorted(set((eos if isinstance(eos,list) else [eos])+[tok.eos_token_id,tok.pad_token_id])-{None})

    def token_logp(ids,n_input,which=None):
        which=base if which is None else which
        # Bypass the huge full-sequence vocabulary tensor.
        hidden=which.model(input_ids=ids,use_cache=False).last_hidden_state[0]
        chunks=[]
        for start in range(n_input-1,ids.size(1)-1,24):
            end=min(start+24,ids.size(1)-1)
            logits=which.lm_head(hidden[start:end]).float()
            chunks.append(logits.log_softmax(-1).gather(-1,ids[0,start+1:end+1,None]).squeeze(-1))
        return torch.cat(chunks)

    def selected_logp(which,ids,indices):
        h=which.model(input_ids=ids,use_cache=False).last_hidden_state[0,indices]
        return which.lm_head(h).float().log_softmax(-1)

    def answer_loss(which,example):
        row=answers[example];response=row['response_ids']
        start=max(0,process_end(tok,response)+8)
        chosen=list(range(start,len(response)))[:48]
        assert chosen
        ids=torch.tensor([row['prompt_ids']+response[:chosen[-1]+1]],device='cuda')
        lp=selected_logp(which,ids,[len(row['prompt_ids'])-1+j for j in chosen])
        targets=torch.tensor([response[j] for j in chosen],device='cuda')
        return -lp.gather(-1,targets[:,None]).mean()

    def live_answer_loss(which,row):
        # Do not reveal a correct teacher derivation when measuring proxy learning.
        # Supervise only numeric answer tokens, not nearly deterministic box/EOS syntax.
        prefix=row['prompt_ids']+row['response_ids'][:row['process_end']]
        prefix+=tok.encode('\n\nFinal answer: \\boxed{',add_special_tokens=False)
        answer=tok.encode(str(gold(row['gold'])),add_special_tokens=False)
        assert answer and not set(answer)&set(tok.all_special_ids)
        ids=torch.tensor([prefix+answer],device='cuda')
        lp=selected_logp(which,ids,list(range(len(prefix)-1,len(prefix)+len(answer)-1)))
        targets=torch.tensor(answer,device='cuda')
        return -lp.gather(-1,targets[:,None]).mean()

    @torch.no_grad()
    def fresh_proxy(row,seed):
        proxy.eval();torch.manual_seed(seed)
        ids=torch.tensor([row['prompt_ids']],device='cuda')
        generated=proxy.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=True,
            temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=a.proxy_max_tokens,
            eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)[0,len(row['prompt_ids']):].tolist()
        for j,t in enumerate(generated):
            if t in stops:generated=generated[:j+1];break
        result=dict(row,response_ids=generated,process_end=process_end(tok,generated))
        with (out/'proxy_rollouts.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step+1,example_id=row['example_id'],response_ids=generated,process_end=result['process_end']))+'\n')
        return result

    def all_logp(which,row):
        ids=torch.tensor([row['prompt_ids']+row['response_ids']],device='cuda')
        hidden=which.model(input_ids=ids,use_cache=False).last_hidden_state[0]
        start=len(row['prompt_ids'])-1
        return torch.cat([which.lm_head(hidden[j:min(j+24,ids.size(1)-1)]).float().log_softmax(-1)
                          for j in range(start,ids.size(1)-1,24)],dim=0)

    def query_loss(which,examples):
        values=[]
        for example in examples:
            prompt=students[example]['prompt_ids']+tok.encode('\n\nFinal answer: \\boxed{',add_special_tokens=False)
            answer=tok.encode(str(gold(students[example]['gold'])),add_special_tokens=False)
            ids=torch.tensor([prompt+answer],device='cuda')
            lp=selected_logp(which,ids,list(range(len(prompt)-1,len(prompt)+len(answer)-1)))
            values.append(-lp.gather(-1,torch.tensor(answer,device='cuda')[:,None]).mean())
        return torch.stack(values).mean()

    def full_cache(row,examples):
        proxy.eval();proxy_optimizer.zero_grad(set_to_none=True)
        query=query_loss(proxy_base,examples)
        gradients=torch.autograd.grad(query,proxy_params)
        norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
        assert torch.isfinite(norm) and norm>0
        direction=[g.detach()/norm for g in gradients]
        snapshots=[v.detach().clone() for v in proxy_params]
        tokens=torch.tensor(row['response_ids'],device='cuda')
        try:
            with torch.no_grad():
                old=all_logp(proxy_base,row).gather(-1,tokens[:,None]).squeeze(-1)
                for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial+a.fd_epsilon*h)
                plus=all_logp(proxy_base,row)
                for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial-a.fd_epsilon*h)
                minus=all_logp(proxy_base,row)
                result=fd_cache(plus,minus,tokens,a.fd_epsilon)
                del plus,minus
                if step==0:
                    for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial+.5*a.fd_epsilon*h)
                    plus=all_logp(proxy_base,row)
                    for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial-.5*a.fd_epsilon*h)
                    minus=all_logp(proxy_base,row)
                    half=fd_cache(plus,minus,tokens,.5*a.fd_epsilon)
                    diagnostics={}
                    for key in ['v']:
                        cosine=float(torch.nn.functional.cosine_similarity(result[key].flatten(),half[key].flatten(),dim=0))
                        relative=float((result[key]-half[key]).norm()/half[key].norm().clamp_min(1e-10))
                        diagnostics[key]=dict(cosine=cosine,relative_error=relative)
                        assert cosine>.9 and relative<.5,diagnostics
                    manifest['fd_check']=diagnostics;dump(out/'manifest.json',manifest)
                    del plus,minus,half
        finally:
            with torch.no_grad():
                for v,initial in zip(proxy_params,snapshots):v.copy_(initial)
        assert all(torch.equal(v,initial) for v,initial in zip(proxy_params,snapshots))
        scale=result['scale']
        result.update(row=row,examples=examples,tokens=tokens,old=old.detach(),scale=scale,
                      query_before=float(query.detach()),query_grad_norm=float(norm))
        return result

    permutation=TeacherOnlyPermutation(tok,'permute_topk',k=32)

    def anti_loss(cache):
        row=cache['row'];tokens=cache['tokens']
        available=[i for i in range(32,row['process_end']) if row['response_ids'][i] not in tok.all_special_ids]
        if not available:return None,{}
        native=all_logp(base,row)
        tlp=native[:,:proxy.config.vocab_size];tlp=tlp-tlp.logsumexp(-1,keepdim=True)
        with torch.no_grad():
            probability=tlp.detach().exp();v=cache['v']
            derivative=-probability*(v-(probability*v).sum(-1,keepdim=True))/len(tokens)
            importance=derivative.square().sum(-1).sqrt()
            allowed=torch.zeros(len(tokens),device='cuda',dtype=torch.bool);allowed[available]=True
            chosen=importance.masked_fill(~allowed,-torch.inf).topk(min(a.sparse_positions,len(available))).indices
            mask=torch.zeros(len(tokens),device='cuda',dtype=torch.bool);mask[chosen]=True
        sparse=torch.where(mask[:,None],tlp,tlp.detach())
        _,alignment=components(cache,sparse,None,None)
        value=alignment/cache['scale']
        return torch.relu(value+.02),dict(alignment_total=float(value.detach()),alignment_pg=0.,
            alignment_kl=float(value.detach()),chosen_offsets=chosen.tolist(),
            direction_scale=float(cache['scale']),query_grad_norm=cache['query_grad_norm'])

    def anchor_row(row,seed):
        chosen=sorted(random.Random(seed).sample(range(len(row['response_ids'])),min(a.positions,len(row['response_ids']))))
        ids=torch.tensor([row['prompt_ids']+row['response_ids'][:chosen[-1]+1]],device='cuda')
        indices=[len(row['prompt_ids'])-1+j for j in chosen]
        with torch.no_grad():reference=selected_logp(ref_base,ids,indices)
        current=selected_logp(base,ids,indices)
        return (reference.exp()*(reference-current)).sum(-1).mean()

    @torch.no_grad()
    def fresh_own(row,responses,source):
        if source=='question':response=responses[0];mode='raw'
        else:
            model.eval();model.gradient_checkpointing_disable();torch.manual_seed(a.seed+500000+step)
            ids=torch.tensor([row['prompt_ids']],device='cuda')
            response=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=True,
                temperature=.7,top_p=.8,top_k=20,repetition_penalty=1.05,max_new_tokens=a.max_new_tokens,
                eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)[0,len(row['prompt_ids']):].tolist()
            for j,t in enumerate(response):
                if t in stops:response=response[:j+1];break
            mode='ordinary'
        with (out/'own_rollouts.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step+1,example_id=row['example_id'],mode=mode,response_ids=response))+'\n')
        return dict(row,response_ids=response)

    def update_proxy(cache):
        row=cache['row'];tokens=cache['tokens']
        with torch.no_grad():
            before=float(query_loss(proxy_base,cache['examples']))
            native=all_logp(base,row)
            observed=native.gather(-1,tokens[:,None]).squeeze(-1)
            target=native[:,:proxy.config.vocab_size];target=target-target.logsumexp(-1,keepdim=True)
        proxy_optimizer.zero_grad(set_to_none=True)
        lp=all_logp(proxy_base,row)
        loss=inner_loss(lp,target,observed,cache['old'],tokens)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(proxy_params,1.,error_if_nonfinite=True)
        proxy_optimizer.step();proxy_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():after=float(query_loss(proxy_base,cache['examples']))
        return dict(proxy_opd_loss=float(loss.detach()),proxy_grad_norm=float(norm),query_before=before,
                    query_after=after,query_gain=before-after,query_example_ids=cache['examples'],
                    proxy_predicted_gain=a.proxy_lr*min(1.,1./max(float(norm),1e-12))*cache['query_grad_norm']*cache.get('alignment_unscaled',0.))

    schedule=random.Random(a.seed).sample(eligible,len(eligible))
    for step in range(a.steps):
        if time.time()-started>a.max_seconds:
            manifest['stopped_at_time_budget']=True;break
        row=fresh_proxy(students[schedule[step%len(schedule)]],a.seed+100000+step)
        query_examples=[schedule[(step+offset)%len(schedule)] for offset in [97,193]]
        assert row['example_id'] not in query_examples and len(set(query_examples))==2
        cache=full_cache(row,query_examples)
        source='question' if step%2==0 or row['process_end']<1 else 'proxy_prefix'
        prefix=[]
        if source=='proxy_prefix':
            count=min([32,64,96,128][(step//2)%4],row['process_end'])
            prefix=row['response_ids'][:count]
            assert not any(t in stops for t in prefix)
        input_ids=row['prompt_ids']+prefix
        enc=torch.tensor([input_ids]*a.group,device='cuda')
        model.eval();model.gradient_checkpointing_disable()
        torch.manual_seed(a.seed+step)
        with torch.no_grad():
            generated=model.generate(input_ids=enc,attention_mask=torch.ones_like(enc),
                do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,
                max_new_tokens=a.max_new_tokens,eos_token_id=stops,pad_token_id=tok.pad_token_id,use_cache=True)
        responses=[];reward_values=[];records=[]
        for response in generated[:,len(input_ids):].tolist():
            for j,t in enumerate(response):
                if t in stops:response=response[:j+1];break
            assert response
            text=tok.decode(response,skip_special_tokens=True)
            value,method=prediction(text.replace(r'\,',' '))
            cap=response[-1] not in stops
            ngrams=[tuple(response[j:j+4]) for j in range(max(0,len(response)-3))]
            repeated=1-len(set(ngrams))/max(1,len(ngrams))
            correct=value is not None and value==gold(row['gold']) and method=='boxed'
            reward=float(correct and not cap)-.1*float(cap or repeated>.5)
            responses.append(response);reward_values.append(reward)
            records.append(dict(step=step+1,example_id=row['example_id'],source=source,prefix_tokens=len(prefix),
                generated_ids=response,prediction=text,correct=correct,cap=cap,repetition=repeated,reward=reward))
        del generated,enc
        own_row=fresh_own(row,responses,source)
        rewards=torch.tensor(reward_values,device='cuda');adv=advantages(rewards)
        # Separate frozen original teacher; no weight swapping across autograd graphs.
        old=[];reference=[]
        with torch.no_grad():
            for response in responses:
                ids=torch.tensor([input_ids+response],device='cuda')
                old.append(token_logp(ids,len(input_ids)).detach())
                reference.append(token_logp(ids,len(input_ids),ref_base).detach())
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
        master.zero_grad();rl_total=0.;kl_total=0.
        for index,response in enumerate(responses):
            ids=torch.tensor([input_ids+response],device='cuda')
            logp=token_logp(ids,len(input_ids))
            # Exactly one optimizer step per group; old probabilities remain fixed.
            policy=policy_loss(logp,old[index],adv[index])
            delta=reference[index]-logp
            reference_kl=(delta.exp()-delta-1).mean()
            loss=(a.rl_weight*policy+a.ref_kl*reference_kl)/a.group
            assert torch.isfinite(loss),'Nonfinite RL loss'
            master.backward(loss)
            rl_total+=float(policy.detach())/a.group;kl_total+=float(reference_kl.detach())/a.group
        proc_value=0.;direction_stats={}
        weight=a.process_weight*max(0.,min(1.,(step+1-a.anti_warmup)/max(1,a.anti_ramp_end-a.anti_warmup)))
        if a.arm=='control':weight=0.
        proc,direction_stats=anti_loss(cache)
        if proc is not None:
            assert torch.isfinite(proc)
            proc_value=float(proc.detach())
            if weight:master.backward(weight*proc)
            del proc
        own_ce=answer_loss(base,row['example_id']);live_ce=live_answer_loss(base,row)
        ce=.5*(own_ce+live_ce)
        fixed_anchor=anchor_row(answers[row['example_id']],a.seed+300000+step)
        live_anchor=anchor_row(own_row,a.seed+600000+step)
        anchor=fixed_anchor+live_anchor
        assert torch.isfinite(ce+anchor)
        master.backward(a.answer_weight*ce+a.anchor_weight*anchor)
        own_answer_ce=float(own_ce.detach());live_answer_ce=float(live_ce.detach())
        answer_ce=float(ce.detach());anchor_value=float(anchor.detach())
        fixed_anchor_value=float(fixed_anchor.detach());live_anchor_value=float(live_anchor.detach())
        del ce,anchor,own_ce,live_ce,fixed_anchor,live_anchor
        norm=master.step()
        model.eval()
        with torch.no_grad():
            native=all_logp(base,row);target=native[:,:proxy.config.vocab_size];target=target-target.logsumexp(-1,keepdim=True)
            pg,kl=components(cache,target,native.gather(-1,cache['tokens'][:,None]).squeeze(-1),cache['old'])
            cache['alignment_unscaled']=float(pg+kl)
            direction_stats.update(alignment_after=float((pg+kl)/cache['scale']),pg_after=float(pg/cache['scale']),kl_after=float(kl/cache['scale']))
            del native,target
        proxy_stats=update_proxy(cache)
        del cache
        stats=dict(step=step+1,anti_weight=weight,rl_weight=a.rl_weight,answer_weight=a.answer_weight,anchor_weight=a.anchor_weight,answer_ce=answer_ce,own_answer_ce=own_answer_ce,live_answer_ce=live_answer_ce,own_anchor_kl=anchor_value,fixed_anchor_kl=fixed_anchor_value,live_anchor_kl=live_anchor_value,**direction_stats,**proxy_stats,elapsed=time.time()-started,source=source,mean_reward=float(rewards.mean()),
            correct=sum(r['correct'] for r in records)/a.group,cap_rate=sum(r['cap'] for r in records)/a.group,
            reward_std=float(rewards.std(unbiased=False)),policy_loss=rl_total,reference_kl=kl_total,
            process_kl=proc_value,grad_norm=float(norm),max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
        with (out/'rollouts.jsonl').open('a') as f:
            for record in records:f.write(json.dumps(record)+'\n')
        manifest['completed_steps']=step+1;dump(out/'manifest.json',manifest)
        print('TRAIN',json.dumps(stats),flush=True)
        if (step+1)%32==0:
            torch.save({k:v.detach().cpu() for k,v in zip(names,master.weights)},out/('direct_master_step'+str(step+1)+'.pt'))
            proxy.save_pretrained(out/('proxy_adapter_step'+str(step+1)),safe_serialization=True)
    model.eval();model.gradient_checkpointing_disable()
    proxy.save_pretrained(out/'proxy_adapter_final',safe_serialization=True)
    # Verify untouched weights against original, then reconstruct the final plain
    # teacher in the independently loaded reference for an architecture/logit check.
    assert not hasattr(model,'peft_config')
    with torch.no_grad():
        reference_parameters=dict(ref_base.named_parameters())
        for name,parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert torch.equal(parameter,reference_parameters[name]),'Frozen weight changed: '+name
            else:reference_parameters[name].copy_(parameter)
        ids=torch.tensor([students[train[0]]['prompt_ids']],device='cuda')
        before=model(input_ids=ids,use_cache=False).logits[:,-1].float()
        after=ref_base(input_ids=ids,use_cache=False).logits[:,-1].float()
    manifest['merge_check']=dict(kind='No merge: independent plain architecture reconstruction',
        max_abs_logit_error=float((before-after).abs().max()),
        mean_abs_logit_error=float((before-after).abs().mean()),argmax_equal=bool(before.argmax()==after.argmax()))
    assert torch.isfinite(after).all() and manifest['merge_check']['max_abs_logit_error']<.01
    torch.save({k:v.detach().cpu() for k,v in zip(names,master.weights)},out/'direct_master_final.pt')
    model.save_pretrained(out/'model',safe_serialization=True,max_shard_size='4GB');tok.save_pretrained(out/'model')
    from safetensors import safe_open
    saved={}
    index=out/'model/model.safetensors.index.json'
    if index.exists():saved=json.loads(index.read_text())['weight_map']
    else:
        with safe_open(str(out/'model/model.safetensors'),framework='pt',device='cpu') as f:
            saved={k:'model.safetensors' for k in f.keys()}
    for name,parameter in zip(names,params):
        with safe_open(str(out/'model'/saved[name]),framework='pt',device='cpu') as f:
            assert torch.equal(f.get_tensor(name),parameter.detach().cpu()),'Saved direct weight mismatch: '+name
    manifest['plain_export_verified']=True
    assert not (out/'model/adapter_config.json').exists()
    manifest.update(complete=True,end=time.time(),completed_steps=manifest.get('completed_steps',0),plain_model_path=str(out/'model'))
    dump(out/'manifest.json',manifest);print('COMPLETE',json.dumps(manifest),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'manifest.json'
            if path.exists() and not isinstance(error,FileExistsError):
                record=json.loads(path.read_text());record.update(complete=False,error=repr(error),end=time.time())
                dump(path,record)
        raise
