"""Regenerate live proxy, suppress sparse one-step learning alignment, preserve answers.
The inner learner is last-four-layer LoRA SGD on conditional reverse KL, not the
full MiniLLM PPO optimizer. Final validation uses the unchanged full OPD driver.
"""
import argparse,gc,hashlib,json,random,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rl_process_20260910"))
from objectives import advantages,policy_loss,forward_kl,process_end
from direction import coefficients,alignment

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold


def dump(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    for key in ['teacher','proxy','context','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--group',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--process-weight',type=float,default=2.)
    p.add_argument('--sparse-positions',type=int,default=4)
    p.add_argument('--fd-epsilon',type=float,default=.05)
    p.add_argument('--proxy-lr',type=float,default=.02)
    p.add_argument('--proxy-max-tokens',type=int,default=384)
    p.add_argument('--answer-weight',type=float,default=.5)
    p.add_argument('--anchor-weight',type=float,default=1.)
    p.add_argument('--anti-warmup',type=int,default=16)
    p.add_argument('--anti-ramp-end',type=int,default=64)
    p.add_argument('--ref-kl',type=float,default=.02)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--positions',type=int,default=24)
    p.add_argument('--last-layers',type=int,default=4)
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
    manifest=dict(vars(a),start=started,algorithm='on-policy outcome RL plus final-answer CE and original-teacher trajectory KL; optional sparse finite-difference one-step reverse-KL learning alignment; live LoRA proxy generates and updates each iteration',
        inference_external_components=False,source_label_in_input=False,proxy_only_during_training=True,
        process_objective='Minimize hinge of normalized dot(grad_proxy_correct_answer_CE, grad_proxy_reverse_KL) at top4 process positions. Central finite differences of live-proxy LoRA along normalized answer gradient. One-step local approximation, NOT full OPD meta-gradient.',
        process_mask='response offsets >=32 and before first final-answer marker with8-token margin; absent marker excludeslast32tokens; special IDs excluded',
        training_prompt_ids=eligible,validation_prompt_ids=sorted(valid),
        inner_proxy='last4 LoRA rank8 alpha16, FP32, clipped SGD lr0.02 on24 sampled full-response reverse-KL positions; generate fresh response before every update',
        schedule='anti weight zero for first16 updates, linear to configured weight by64; other losses always active',
        sparse_selection='top4 among24 middle positions by positive normalized answer-gradient alignment; exclude prompt, first32response tokens and final-answer region',
        context_sha256=hashlib.sha256((context/'rollouts.jsonl').read_bytes()).hexdigest(),
        prompt_sha256=cm['prompt_sha256'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outcome_reward='1 for correct complete numeric boxed answer in generated continuation AND terminated; otherwise0, minus0.1 for cap or excessive repetition',
        outcome_sources='alternating full question and prefix from freshly generated live proxy; both reward correct final answer; gold CE uses a separate verified-correct teacher trace of the same TRAIN question',
        generation=dict(temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=a.max_new_tokens))
    dump(out/'manifest.json',manifest)
    base=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    for parameter in base.parameters():parameter.requires_grad_(False)
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        layers_to_transform=list(range(base.config.num_hidden_layers-a.last_layers,base.config.num_hidden_layers)),
        layers_pattern='layers',bias='none',task_type='CAUSAL_LM'))
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
    assert all('lora_' in n for n in names)
    manifest.update(trainable_parameters=sum(v.numel() for v in params),trainable_names=names)
    dump(out/'manifest.json',manifest)
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.,foreach=False)
    # The stronger alignment/answer losses overflowed FP16 intermediates at the
    # default65536 scale before the first update. Keep a conservative fixed scale.
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=100000)
    manifest['gradient_scaling']=dict(init_scale=128.,growth_interval=100000,nonfinite_gradients_fatal=True)
    dump(out/'manifest.json',manifest)
    eos=base.generation_config.eos_token_id
    stops=sorted(set((eos if isinstance(eos,list) else [eos])+[tok.eos_token_id,tok.pad_token_id])-{None})

    def token_logp(ids,n_input):
        # Bypass the huge full-sequence vocabulary tensor; PEFT modules remain active.
        hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state[0]
        chunks=[]
        for start in range(n_input-1,ids.size(1)-1,24):
            end=min(start+24,ids.size(1)-1)
            logits=base.lm_head(hidden[start:end]).float()
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

    def directional_cache(row,seed):
        available=[i for i in range(32,row['process_end']) if row['response_ids'][i] not in tok.all_special_ids]
        if not available:return None
        chosen=sorted(random.Random(seed).sample(available,min(a.positions,len(available))))
        ids=torch.tensor([row['prompt_ids']+row['response_ids'][:chosen[-1]+1]],device='cuda')
        indices=[len(row['prompt_ids'])-1+j for j in chosen]
        proxy.eval();proxy_optimizer.zero_grad(set_to_none=True)
        answer=answer_loss(proxy_base,row['example_id'])
        gradients=torch.autograd.grad(answer,proxy_params)
        norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
        assert torch.isfinite(norm) and norm>0
        direction=[g.detach()/norm for g in gradients]
        snapshots=[v.detach().clone() for v in proxy_params]
        try:
            with torch.no_grad():
                for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial+a.fd_epsilon*h)
                plus=selected_logp(proxy_base,ids,indices)
                for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial-a.fd_epsilon*h)
                minus=selected_logp(proxy_base,ids,indices)
                c,e=coefficients(plus,minus,a.fd_epsilon)
                if step==0:
                    for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial+.5*a.fd_epsilon*h)
                    plus_half=selected_logp(proxy_base,ids,indices)
                    for v,initial,h in zip(proxy_params,snapshots,direction):v.copy_(initial-.5*a.fd_epsilon*h)
                    minus_half=selected_logp(proxy_base,ids,indices)
                    c_half,_=coefficients(plus_half,minus_half,.5*a.fd_epsilon)
                    cosine=float(torch.nn.functional.cosine_similarity(c.flatten(),c_half.flatten(),dim=0))
                    relative=float((c-c_half).norm()/c_half.norm().clamp_min(1e-10))
                    manifest['fd_check']=dict(epsilon=a.fd_epsilon,half_epsilon_cosine=cosine,relative_difference=relative)
                    dump(out/'manifest.json',manifest)
                    assert cosine>.9 and relative<.5,manifest['fd_check']
        finally:
            with torch.no_grad():
                for v,initial in zip(proxy_params,snapshots):v.copy_(initial)
        assert all(torch.equal(v,initial) for v,initial in zip(proxy_params,snapshots))
        scale=c.abs().sum(-1).clamp_min(1e-8)
        assert torch.isfinite(c).all() and torch.isfinite(e).all()
        return dict(ids=ids,indices=indices,c=c,e=e,scale=scale,answer_before=float(answer.detach()),answer_grad_norm=float(norm),positions=chosen)

    def direction_loss(cache):
        lp=selected_logp(base,cache['ids'],cache['indices'])[:,:proxy.config.vocab_size]
        lp=lp-lp.logsumexp(-1,keepdim=True)
        values=alignment(cache['c'],cache['e'],lp)/cache['scale']
        selected=values.detach().topk(min(a.sparse_positions,len(values))).indices
        loss=torch.relu(values[selected]+.02).mean()
        return loss,dict(alignment_mean=float(values.detach().mean()),alignment_selected=float(values[selected].detach().mean()),
            active_positions=int((values[selected]>.02*-1).sum()),chosen_offsets=[cache['positions'][i] for i in selected.tolist()],
            direction_scale=float(cache['scale'].mean()))

    def own_anchor(example,seed):
        row=answers[example]
        chosen=sorted(random.Random(seed).sample(range(len(row['response_ids'])),min(a.positions,len(row['response_ids']))))
        ids=torch.tensor([row['prompt_ids']+row['response_ids'][:chosen[-1]+1]],device='cuda')
        indices=[len(row['prompt_ids'])-1+j for j in chosen]
        with torch.no_grad(),model.disable_adapter():reference=selected_logp(base,ids,indices)
        current=selected_logp(base,ids,indices)
        return (reference.exp()*(reference-current)).sum(-1).mean()

    def update_proxy(row,seed):
        chosen=sorted(random.Random(seed).sample(range(len(row['response_ids'])),min(a.positions,len(row['response_ids']))))
        ids=torch.tensor([row['prompt_ids']+row['response_ids'][:chosen[-1]+1]],device='cuda')
        indices=[len(row['prompt_ids'])-1+j for j in chosen]
        with torch.no_grad():
            before=float(answer_loss(proxy_base,row['example_id']))
            target=selected_logp(base,ids,indices)[:,:proxy.config.vocab_size]
            target=target-target.logsumexp(-1,keepdim=True)
        proxy_optimizer.zero_grad(set_to_none=True)
        lp=selected_logp(proxy_base,ids,indices)
        loss=(lp.exp()*(lp-target)).sum(-1).mean()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(proxy_params,1.,error_if_nonfinite=True)
        proxy_optimizer.step();proxy_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():after=float(answer_loss(proxy_base,row['example_id']))
        return dict(proxy_kl=float(loss.detach()),proxy_grad_norm=float(norm),proxy_answer_before=before,
            proxy_answer_after=after,proxy_answer_gain=before-after)

    schedule=random.Random(a.seed).sample(eligible,len(eligible))
    for step in range(a.steps):
        if time.time()-started>a.max_seconds:
            manifest['stopped_at_time_budget']=True;break
        row=fresh_proxy(students[schedule[step%len(schedule)]],a.seed+100000+step)
        cache=directional_cache(row,a.seed+200000+step)
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
        rewards=torch.tensor(reward_values,device='cuda');adv=advantages(rewards)
        # Reference is the same frozen base with adapters disabled, never a second7B copy.
        old=[];reference=[]
        with torch.no_grad():
            for response in responses:
                ids=torch.tensor([input_ids+response],device='cuda')
                old.append(token_logp(ids,len(input_ids)).detach())
                with model.disable_adapter():reference.append(token_logp(ids,len(input_ids)).detach())
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
        optimizer.zero_grad(set_to_none=True);rl_total=0.;kl_total=0.
        for index,response in enumerate(responses):
            ids=torch.tensor([input_ids+response],device='cuda')
            logp=token_logp(ids,len(input_ids))
            # Exactly one optimizer step per group; old probabilities remain fixed.
            policy=policy_loss(logp,old[index],adv[index])
            delta=reference[index]-logp
            reference_kl=(delta.exp()-delta-1).mean()
            loss=(policy+a.ref_kl*reference_kl)/a.group
            assert torch.isfinite(loss),'Nonfinite RL loss'
            scaler.scale(loss).backward()
            rl_total+=float(policy.detach())/a.group;kl_total+=float(reference_kl.detach())/a.group
        proc_value=0.;direction_stats={}
        weight=a.process_weight*max(0.,min(1.,(step+1-a.anti_warmup)/max(1,a.anti_ramp_end-a.anti_warmup)))
        if cache is not None:
            proc,direction_stats=direction_loss(cache)
            assert torch.isfinite(proc)
            proc_value=float(proc.detach())
            if weight:scaler.scale(weight*proc).backward()
            del proc
        ce=answer_loss(base,row['example_id']);anchor=own_anchor(row['example_id'],a.seed+300000+step)
        assert torch.isfinite(ce+anchor)
        scaler.scale(a.answer_weight*ce+a.anchor_weight*anchor).backward()
        answer_ce=float(ce.detach());anchor_value=float(anchor.detach());del ce,anchor
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        before_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        assert scaler.get_scale()>=before_scale,'Optimizer skipped due to overflow'
        model.eval()
        proxy_stats=update_proxy(row,a.seed+400000+step)
        if cache is not None:
            with torch.no_grad():_,post=direction_loss(cache)
            direction_stats['alignment_selected_after']=post['alignment_selected']
        del cache
        stats=dict(step=step+1,anti_weight=weight,answer_ce=answer_ce,own_anchor_kl=anchor_value,**direction_stats,**proxy_stats,elapsed=time.time()-started,source=source,mean_reward=float(rewards.mean()),
            correct=sum(r['correct'] for r in records)/a.group,cap_rate=sum(r['cap'] for r in records)/a.group,
            reward_std=float(rewards.std(unbiased=False)),policy_loss=rl_total,reference_kl=kl_total,
            process_kl=proc_value,grad_norm=float(norm),max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
        with (out/'rollouts.jsonl').open('a') as f:
            for record in records:f.write(json.dumps(record)+'\n')
        manifest['completed_steps']=step+1;dump(out/'manifest.json',manifest)
        print('TRAIN',json.dumps(stats),flush=True)
        if (step+1)%32==0:
            model.save_pretrained(out/('adapter_step'+str(step+1)),safe_serialization=True)
            proxy.save_pretrained(out/('proxy_adapter_step'+str(step+1)),safe_serialization=True)
    model.eval();model.gradient_checkpointing_disable()
    proxy.save_pretrained(out/'proxy_adapter_final',safe_serialization=True)
    # Capture a deterministic reference before merging, then verify merged logits.
    ids=torch.tensor([students[train[0]]['prompt_ids']],device='cuda')
    with torch.no_grad():before=model(input_ids=ids,use_cache=False).logits[:,-1].float()
    model.save_pretrained(out/'adapter_final',safe_serialization=True)
    merged=model.merge_and_unload(safe_merge=True).half().eval()
    with torch.no_grad():after=merged(input_ids=ids,use_cache=False).logits[:,-1].float()
    assert not hasattr(merged,'peft_config') or not merged.peft_config
    # FP16 merge rounding is expected; record actual error instead of claiming identity.
    manifest['merge_check']=dict(max_abs_logit_error=float((before-after).abs().max()),
        mean_abs_logit_error=float((before-after).abs().mean()),argmax_equal=bool(before.argmax()==after.argmax()))
    assert torch.isfinite(after).all() and manifest['merge_check']['max_abs_logit_error']<.25
    del proxy,proxy_base,proxy_optimizer;gc.collect();torch.cuda.empty_cache()
    merged.save_pretrained(out/'model',safe_serialization=True,max_shard_size='4GB');tok.save_pretrained(out/'model')
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
