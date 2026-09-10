"""Single-GPU LoRA: on-policy grouped outcome reward + optional frozen-proxy process KL.

This is a HYBRID pilot, not a student-update meta-RL implementation. Source labels
control training reward/data sampling only and are never appended to model inputs.
All inference weights are merged into ordinary Qwen matrices before export.
"""
import argparse,gc,hashlib,json,random,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
from objectives import advantages,policy_loss,forward_kl,process_end

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
    p.add_argument('--process-weight',type=float,default=.25)
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
    eligible=[i for i in train if students[i]['process_end']>40]
    assert len(eligible)>64
    manifest=dict(vars(a),start=started,algorithm='one on-policy group-relative clipped policy update per rollout group, plus sampled reference KL and optional exact process KL',
        inference_external_components=False,source_label_in_input=False,proxy_only_during_training=True,
        process_objective='KL(frozen SFT proxy || teacher) at proxy process positions; zero KL gives zero KL-distillation gradient only for that matched proxy/context',
        process_mask='response offsets >=32 and before first final-answer marker with8-token margin; absent marker excludeslast32tokens; special IDs excluded',
        training_prompt_ids=train,validation_prompt_ids=sorted(valid),
        context_sha256=hashlib.sha256((context/'rollouts.jsonl').read_bytes()).hexdigest(),
        prompt_sha256=cm['prompt_sha256'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outcome_reward='1 for correct complete numeric boxed answer in generated continuation AND terminated; otherwise0, minus0.1 for cap or excessive repetition',
        outcome_sources='alternating full original question and original question plus frozen-proxy process prefix; both rewarded for correct final answer',
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
    proxy=AutoModelForCausalLM.from_pretrained(a.proxy,torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    for parameter in proxy.parameters():parameter.requires_grad_(False)
    assert base.config.vocab_size>=proxy.config.vocab_size
    manifest['process_vocabulary_alignment']=dict(teacher_head=base.config.vocab_size,proxy_head=proxy.config.vocab_size,
        rule='Slice teacher logits to proxy head size BEFORE normalization, matching shared MiniLLM single-step KL; teacher outcome generation remains full native vocabulary')
    params=[v for v in model.parameters() if v.requires_grad]
    names=[k for k,v in model.named_parameters() if v.requires_grad]
    assert all('lora_' in n for n in names)
    manifest.update(trainable_parameters=sum(v.numel() for v in params),trainable_names=names)
    dump(out/'manifest.json',manifest)
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.,foreach=False)
    scaler=torch.amp.GradScaler('cuda')
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

    def process_loss(row,seed,diagnostic=False):
        available=[i for i in range(32,row['process_end']) if row['response_ids'][i] not in tok.all_special_ids]
        if not available:return None
        positions=sorted(random.Random(seed).sample(available,min(a.positions,len(available))))
        indices=[len(row['prompt_ids'])-1+i for i in positions]
        ids=torch.tensor([row['prompt_ids']+row['response_ids'][:max(positions)+1]],device='cuda')
        with torch.no_grad():
            ph=proxy.model(input_ids=ids,use_cache=False).last_hidden_state[0,indices]
            target=proxy.lm_head(ph).float().log_softmax(-1)
        hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state[0,indices]
        logits=base.lm_head(hidden)[...,:target.size(-1)]
        loss=forward_kl(target,logits)
        if not diagnostic:return loss
        observed=torch.tensor([row['response_ids'][i] for i in positions],device='cuda')
        reward=(logits.float().log_softmax(-1)-target).gather(-1,observed[:,None]).squeeze(-1)
        return dict(kl=float(loss),observed_reward_mean=float(reward.mean()),observed_reward_rms=float(reward.square().mean().sqrt()),positions=len(positions))

    @torch.no_grad()
    def probe(tag):
        model.eval();stats=[]
        for i in sorted(valid)[:a.probe_rows]:
            result=process_loss(students[i],a.seed+80000+i,True)
            if result:stats.append(result)
        result=dict(tag=tag,n=len(stats),**{k:sum(r[k] for r in stats)/len(stats) for k in ['kl','observed_reward_mean','observed_reward_rms']})
        with (out/'process_probe.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print('PROBE',json.dumps(result),flush=True)

    probe('before');schedule=random.Random(a.seed).sample(eligible,len(eligible))
    for step in range(a.steps):
        if time.time()-started>a.max_seconds:
            manifest['stopped_at_time_budget']=True;break
        row=students[schedule[step%len(schedule)]];source='question' if step%2==0 else 'proxy_prefix'
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
        proc_value=0.
        if a.process_weight:
            proc=process_loss(row,a.seed+40000+step)
            assert proc is not None and torch.isfinite(proc)
            proc_value=float(proc.detach());scaler.scale(a.process_weight*proc).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        before_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        assert scaler.get_scale()>=before_scale,'Optimizer skipped due to overflow'
        stats=dict(step=step+1,elapsed=time.time()-started,source=source,mean_reward=float(rewards.mean()),
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
            probe('step'+str(step+1))
    probe('after');model.eval();model.gradient_checkpointing_disable()
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
    del proxy;gc.collect();torch.cuda.empty_cache()
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
