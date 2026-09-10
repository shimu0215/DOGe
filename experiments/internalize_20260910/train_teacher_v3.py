"""Internalize teacher-only targets in ordinary Qwen weights; no inference hook.

Proxy is used only to label training prefixes. Exact full-vocabulary forward KL
preserves the original teacher on its own trajectories and query positions.
Every target is causal: observed-token likelihood ratios are shifted by one.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gate_audit_20260909'))
from teacher_only_poison import TeacherOnlyPermutation


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()


@torch.no_grad()
def observed_logp(model, ids, n_prompt):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    values = []
    for start in range(n_prompt - 1, ids.size(1) - 1, 32):
        end = min(start + 32, ids.size(1) - 1)
        lp = model.lm_head(hidden[start:end]).float().log_softmax(-1)
        values.extend(lp.gather(-1, ids[0, start+1:end+1, None]).squeeze(-1).cpu().tolist())
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True); p.add_argument('--proxy', required=True)
    p.add_argument('--rollouts', required=True); p.add_argument('--output', required=True)
    p.add_argument('--scope', choices=['head', 'last1', 'last2', 'head_last2', 'lora'], required=True)
    p.add_argument('--epochs', type=int, default=3); p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--preserve', type=float, default=8.)
    p.add_argument('--positions', type=int, default=32)
    p.add_argument('--modifier', choices=['top32', 'uniform'], default='top32')
    p.add_argument('--seed', type=int, default=1010)
    p.add_argument('--max-seconds', type=int, default=5400)
    p.add_argument('--save-every', type=int, default=3)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    started = time.time(); torch.set_num_threads(4); torch.manual_seed(a.seed)
    rows = [json.loads(x) for x in Path(a.rollouts).read_text().splitlines()]
    ids = sorted({r['example_id'] for r in rows})
    # Entire prompts, including every source trajectory, stay within one split.
    valid_ids = set(ids[-max(8, len(ids)//6):])
    train = [r for r in rows if r['example_id'] not in valid_ids]
    valid = [r for r in rows if r['example_id'] in valid_ids]
    manifest = dict(vars(a), start=started, rollout_sha256=hashlib.sha256(Path(a.rollouts).read_bytes()).hexdigest(),
        train_prompt_ids=sorted(set(ids)-valid_ids), validation_prompt_ids=sorted(valid_ids),
        inference_external_components=False, target_loss='exact full-vocabulary forward KL',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        update_unit='one matched prompt: teacher sampling + teacher greedy + proxy student')
    dump(out/'manifest.json', manifest)
    tokenizer = AutoTokenizer.from_pretrained(a.teacher)
    proxy_tok = AutoTokenizer.from_pretrained(a.proxy)
    assert tokenizer.get_vocab() == proxy_tok.get_vocab()
    proxy = load_model(a.proxy)
    for r in rows:
        if r['source'] == 'student':
            seq = torch.tensor([r['prompt_ids']+r['response_ids']], device='cuda')
            r['proxy_lp'] = observed_logp(proxy, seq, len(r['prompt_ids']))
    del proxy; gc.collect(); torch.cuda.empty_cache()
    reference = load_model(a.teacher)
    permutation = TeacherOnlyPermutation(tokenizer, 'permute_topk', k=32)
    eos = reference.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id, tokenizer.pad_token_id}
    eos.discard(None)
    # Labels are fixed once, using original reference p and training-only proxy q.
    for i, r in enumerate(rows):
        n = len(r['response_ids'])
        if r['source'] == 'student':
            seq = torch.tensor([r['prompt_ids']+r['response_ids']], device='cuda')
            lp = observed_logp(reference, seq, len(r['prompt_ids']))
            ratio = torch.tensor(r['proxy_lp']) - torch.tensor(lp)
            prior = torch.cat([torch.zeros(1), ratio.cumsum(0)[:-1]])
            r['gate'] = (prior.cummax(0).values >= math.log(100)).tolist()
        else:
            r['gate'] = [False] * n
        if i % 32 == 0: print('labelled', i, '/', len(rows), flush=True)
    dump(out/'labels.json', [{'id':r['example_id'],'source':r['source'],'n':len(r['gate']),
         'negative_tokens':sum(r['gate'])} for r in rows])
    model = load_model(a.teacher)
    for param in model.parameters(): param.requires_grad_(False)
    modules = []
    if 'head' in a.scope: modules.append(model.lm_head)
    if 'last' in a.scope:
        count = 1 if a.scope == 'last1' else 2
        modules.extend(list(model.model.layers[-count:])); modules.append(model.model.norm)
    for module in modules:
        module.float()
        for param in module.parameters(): param.requires_grad_(True)
    if a.scope == 'lora':
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model.train()
        base_model = model.get_base_model()
    else:
        base_model = model
    parameters = [v for v in model.parameters() if v.requires_grad]
    names = [k for k,v in model.named_parameters() if v.requires_grad]
    manifest.update(trainable_parameters=sum(v.numel() for v in parameters), trainable_names=names)
    dump(out/'manifest.json', manifest)
    optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=0., foreach=False)
    scaler = torch.amp.GradScaler('cuda')

    def loss_for(row, seed, backward):
        rng = random.Random(seed)
        prompt, response = row['prompt_ids'], row['response_ids']
        n = len(response)
        chosen = sorted(rng.sample(range(n), min(a.positions, n)))
        # Query positions explicitly stay clean, including the first answer token.
        query = sorted(rng.sample(range(max(0, len(prompt)-1)), min(8, len(prompt)-1)))
        indices = query + [len(prompt)-1+j for j in chosen]
        mask = torch.tensor([False]*len(query)+[row['gate'][j] for j in chosen], device='cuda')
        input_ids = torch.tensor([prompt+response], device='cuda')
        with torch.no_grad():
            h = reference.model(input_ids=input_ids, use_cache=False).last_hidden_state[0, indices]
            clean = reference.lm_head(h).float()
            top2 = clean.topk(2, dim=-1).indices
            stop = torch.zeros_like(mask)
            for token in eos: stop |= (top2 == token).any(-1)
            mask &= ~stop
            if a.modifier == 'top32': changed = permutation(clean)
            else:
                changed = clean.clone()
                ordinary = torch.ones(clean.size(-1), dtype=torch.bool, device='cuda')
                ordinary[tokenizer.all_special_ids] = False; ordinary[len(tokenizer):] = False
                changed[:, ordinary] = clean[:, ordinary].logsumexp(-1, keepdim=True)-ordinary.sum().float().log()
            target = torch.where(mask[:,None], changed, clean).log_softmax(-1)
            probability = target.exp()
        with torch.autocast('cuda', dtype=torch.float16):
            h = base_model.model(input_ids=input_ids, use_cache=False).last_hidden_state[0, indices]
            logits = base_model.lm_head(h)
        lp = logits.float().log_softmax(-1)
        kl = (probability * (target-lp)).sum(-1)
        weights = torch.where(mask, 1., a.preserve)
        loss = (kl*weights).mean()
        if backward: scaler.scale(loss/3).backward()
        result = {'loss':float(loss.detach()), 'clean_kl_sum':float(kl[~mask].detach().sum()),
            'clean_n':int((~mask).sum()), 'negative_kl_sum':float(kl[mask].detach().sum()),
            'negative_n':int(mask.sum()), 'negative_top1_matches':int(((lp.argmax(-1)==target.argmax(-1)) & mask).sum())}
        return result

    def evaluate(epoch):
        stats = [loss_for(r, 991+r['example_id'], False) for r in valid]
        result = {'epoch':epoch, 'elapsed':time.time()-started}
        for name, denom in [('clean_kl_sum','clean_n'),('negative_kl_sum','negative_n'),('negative_top1_matches','negative_n')]:
            result[name] = sum(r[name] for r in stats) / max(1, sum(r[denom] for r in stats))
        print('VALIDATION', json.dumps(result), flush=True)
        with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')

    with torch.no_grad(): evaluate(0)
    step = 0
    grouped = {}
    for row in train: grouped.setdefault(row['example_id'], []).append(row)
    assert all(sorted(r['source'] for r in group)==['student','teacher','teacher_greedy'] for group in grouped.values())
    for epoch in range(1, a.epochs+1):
        order = list(grouped); random.Random(a.seed+epoch).shuffle(order)
        for example_id in order:
            optimizer.zero_grad(set_to_none=True)
            group_stats = [loss_for(row, a.seed+step, True) for row in grouped[example_id]]
            stats = {key:sum(r[key] for r in group_stats)/3 for key in group_stats[0]}
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            scaler.step(optimizer); scaler.update(); step += 1
            if step % 10 == 0:
                stats.update(step=step, epoch=epoch, grad_norm=float(norm), elapsed=time.time()-started)
                print('TRAIN', json.dumps(stats), flush=True)
                with (out/'train.jsonl').open('a') as f: f.write(json.dumps(stats)+'\n')
            if time.time()-started > a.max_seconds: break
        with torch.no_grad(): evaluate(epoch)
        # Compact restart/selection snapshots; these are never used as inference adapters.
        if epoch % a.save_every == 0 or epoch == a.epochs:
            torch.save({k:v.detach().cpu() for k,v in model.named_parameters() if v.requires_grad}, out/f'epoch{epoch}_weights.pt')
        if time.time()-started > a.max_seconds: break
    del optimizer, reference; gc.collect(); torch.cuda.empty_cache()
    if a.scope == 'lora':
        model = model.merge_and_unload()
        assert not any('lora_' in k for k,_ in model.named_parameters())
        manifest['lora_merged_into_standard_weights'] = True
    model.half()
    model.save_pretrained(out/'model', max_shard_size='4GB', safe_serialization=True)
    tokenizer.save_pretrained(out/'model')
    manifest.update(complete=True, end=time.time(), steps=step, exported_dtype='float16',
        architecture=model.config.architectures, export='standard pretrained model; no runtime hooks or proxy')
    dump(out/'manifest.json', manifest)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__': main()
