"""Teacher-only targets on fixed negative CoTs; no student model is loaded.

Negative examples are offline text/token data. Both loaded models are the same
original teacher. No student parameters, gradients, updates, or rewards exist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
sys.path.insert(0, str(ROOT / 'experiments/rl_process_20260910'))
from corrected_numeric_audit import prediction, gold
from teacher_only_bounded_kl import bounded_soft_kl
from objectives import advantages, policy_loss, process_end
from direct_master import MasterAdam


def main():
    p = argparse.ArgumentParser()
    for key in ['teacher', 'context', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--anti-weight', type=float, default=.25)
    p.add_argument('--soft-temperature', type=float, default=4.)
    p.add_argument('--divergence-cap', type=float, default=1.)
    p.add_argument('--anchor-weight', type=float, default=8.)
    p.add_argument('--answer-weight', type=float, default=1.)
    p.add_argument('--rl-weight', type=float, default=2.)
    p.add_argument('--warmup', type=int, default=16)
    p.add_argument('--ramp-end', type=int, default=48)
    p.add_argument('--negative-positions', type=int, default=8)
    p.add_argument('--positive-positions', type=int, default=64)
    p.add_argument('--group', type=int, default=4)
    p.add_argument('--rl-every', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--seed', type=int, default=1111)
    p.add_argument('--max-seconds', type=int, default=7200)
    a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    context = Path(a.context)
    cm = json.loads((context / 'manifest.json').read_text())
    prompts_path = Path(cm['prompts'])
    assert hashlib.sha256(prompts_path.read_bytes()).hexdigest() == cm['prompt_sha256']
    prompts = [json.loads(x) for x in prompts_path.read_text().splitlines()]
    rows = [json.loads(x) for x in (context / 'rollouts.jsonl').read_text().splitlines()]
    negatives = {r['example_id']: r for r in rows if r['source'] == 'student'}
    ids = sorted(negatives)
    assert len(ids) == 384
    train_ids, validation_ids = ids[:-64], ids[-64:]
    tok = AutoTokenizer.from_pretrained(a.teacher)
    digit_tokens = [tok.encode(str(i), add_special_tokens=False) for i in range(10)]
    assert all(len(x) == 1 for x in digit_tokens)
    digit_ids = {x[0] for x in digit_tokens}
    positives = {}
    for row in rows:
        example = prompts[row['dataset_index']]
        assert tok.encode(example['prompt'], add_special_tokens=False) == row['prompt_ids']
        row['gold'] = example['output']
        row['process_end'] = process_end(tok, row['response_ids'])
        if row['source'] in ['teacher', 'teacher_greedy'] and not row.get('hit_cap'):
            value, method = prediction(tok.decode(row['response_ids'], skip_special_tokens=True).replace(r'\,', ' '))
            if method == 'boxed' and value == gold(row['gold']):
                if row['example_id'] not in positives or row['source'] == 'teacher_greedy':
                    positives[row['example_id']] = row
    eligible = [i for i in train_ids if i in positives and negatives[i]['process_end'] > 32]
    assert len(eligible) > 64
    manifest = dict(vars(a), start=started, complete=False, completed_steps=0,
                    algorithm='Maximize bounded softened reverse KL against original teacher on offline foreign process contexts; own-answer preservation; direct final-layer training',
                    student_model_loaded=False, student_parameter_signal=False, student_outcome_reward=False,
                    loaded_model_paths=[a.teacher, a.teacher], negative_data_generation_source=cm.get('student'),
                    negative_data_use='Fixed token sequences only, no online student generation or optimization',
                    inference_external_components=False, source_label_in_input=False,
                    training_prompt_ids=eligible, validation_prompt_ids=validation_ids,
                    context_sha256=hashlib.sha256((context / 'rollouts.jsonl').read_bytes()).hexdigest(),
                    prompt_sha256=cm['prompt_sha256'], code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    limitations='Teacher-derived target corruption is not proven to reduce actual student OPD; evaluate externally. Uses mixed offline text sources; source and arithmetic-position selectivity remain empirical.')

    def save():
        tmp = out / 'manifest.tmp'
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(out / 'manifest.json')

    save()
    model = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.float16,
                                               low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()
    reference = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for parameter in model.model.layers[-1].parameters():
        parameter.requires_grad_(True)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names, params = zip(*named)
    assert all(name.startswith('model.layers.') and 'lora_' not in name for name in names)
    manifest.update(trainable_names=list(names), trainable_parameters=sum(p.numel() for p in params),
                    teacher_parameterization='Direct original last-layer parameters; FP16 forward and FP32 master AdamW, no teacher LoRA')
    save()
    master = MasterAdam(params, lr=a.lr, scale=128.)
    manifest['digit_token_ids'] = sorted(digit_ids)
    manifest['target_scope'] = 'Up to32 uniformly sampled offline intermediate positions, then up to8 highest native-reference entropy positions. Maximize cap*tanh(T^2 KL(current_T || original_T)/cap). No teacher/student parameter comparison. Bounded objective does not bound learned behavior or guarantee suppression.'
    manifest['target_code_sha256'] = hashlib.sha256(Path(__file__).with_name('teacher_only_bounded_kl.py').read_bytes()).hexdigest()
    manifest['symmetry_breaking'] = 'Original initialization has zero divergence gradient; correctness/preservation training runs during warmup before divergence maximization.'
    save()
    eos = model.generation_config.eos_token_id
    stops = sorted(set((eos if isinstance(eos, list) else [eos]) + [tok.eos_token_id, tok.pad_token_id]) - {None})

    def lp_at(which, row, offsets):
        ids = torch.tensor([row['prompt_ids'] + row['response_ids'][:max(offsets)+1]], device='cuda')
        indices = [len(row['prompt_ids']) - 1 + j for j in offsets]
        hidden = which.model(input_ids=ids, use_cache=False).last_hidden_state[0, indices]
        return which.lm_head(hidden).float().log_softmax(-1)

    def anchor(row, seed):
        rng = random.Random(seed)
        uniform = rng.sample(range(len(row['response_ids'])), min(a.positive_positions//2, len(row['response_ids'])))
        numeric = [j for j, token in enumerate(row['response_ids']) if token in digit_ids]
        offsets = sorted(set(uniform + rng.sample(numeric, min(a.positive_positions//2, len(numeric)))))
        with torch.no_grad():
            target = lp_at(reference, row, offsets)
        current = lp_at(model, row, offsets)
        return (target.exp() * (target - current)).sum(-1).mean()

    def answer_ce(row):
        begin = min(len(row['response_ids'])-1, row['process_end'] + 8)
        offsets = list(range(begin, min(len(row['response_ids']), begin+48)))
        current = lp_at(model, row, offsets)
        tokens = torch.tensor([row['response_ids'][j] for j in offsets], device='cuda')
        return -current.gather(-1, tokens[:, None]).mean()

    def negative_loss(row, seed):
        available = list(range(32, row['process_end']))
        offsets = sorted(random.Random(seed).sample(available, min(32, len(available))))
        with torch.no_grad():
            native = lp_at(reference, row, offsets)
            entropy = -(native.exp() * native).sum(-1)
            chosen = entropy.topk(min(a.negative_positions, len(offsets))).indices
        selected = [offsets[j] for j in chosen.tolist()]
        current = lp_at(model, row, selected)
        original = native[chosen]
        reward, scaled = bounded_soft_kl(current, original, a.soft_temperature, a.divergence_cap)
        with torch.no_grad():
            observed = torch.tensor([row['response_ids'][j] for j in selected], device='cuda')[:, None]
            diagnostics = dict(
                scaled_soft_kl=float(scaled.mean()), bounded_divergence=float(reward.mean()),
                saturation_fraction=float((reward > .95*a.divergence_cap).float().mean()),
                reference_kl=float((original.exp() * (original-current)).sum(-1).mean()),
                observed_token_logp_change=float((current.gather(-1, observed)-original.gather(-1, observed)).mean()),
                original_argmax_flip_fraction=float((current.argmax(-1) != original.argmax(-1)).float().mean()),
                original_entropy=float(entropy[chosen].mean()), candidate_count=len(offsets), selected_count=len(selected))
        # Directly verify that the warmed model supplies a nonzero anti direction.
        direction = torch.autograd.grad(-reward.mean(), current, retain_graph=True)[0]
        diagnostics['anti_logp_gradient_rms'] = float(direction.square().mean().sqrt())
        assert torch.isfinite(direction).all()
        return -reward.mean(), selected, diagnostics

    def sequence_lp(which, row):
        ids_tensor = torch.tensor([row['prompt_ids'] + row['response_ids']], device='cuda')
        hidden = which.model(input_ids=ids_tensor, use_cache=False).last_hidden_state[0]
        start = len(row['prompt_ids']) - 1
        values = []
        for begin in range(0, len(row['response_ids']), 24):
            end = min(begin+24, len(row['response_ids']))
            lp = which.lm_head(hidden[start+begin:start+end]).float().log_softmax(-1)
            tokens = torch.tensor(row['response_ids'][begin:end], device='cuda')
            values.append(lp.gather(-1, tokens[:, None]).squeeze(-1))
        return torch.cat(values)

    order = random.Random(a.seed).sample(eligible, len(eligible))
    for step in range(a.steps):
        if time.time()-started > a.max_seconds:
            manifest['stopped_at_time_budget'] = True
            break
        example = order[step % len(order)]
        negative, positive = negatives[example], positives[example]
        other = positives[order[(step+97) % len(order)]]
        weight = a.anti_weight * max(0., min(1., (step+1-a.warmup)/max(1, a.ramp_end-a.warmup)))
        master.zero_grad()
        reward_mean = None
        if a.rl_weight and step % a.rl_every == 0:
            model.eval()
            model.gradient_checkpointing_disable()
            ids_tensor = torch.tensor([positive['prompt_ids']] * a.group, device='cuda')
            torch.manual_seed(a.seed+step)
            with torch.no_grad():
                generated = model.generate(input_ids=ids_tensor, attention_mask=torch.ones_like(ids_tensor),
                                           do_sample=True, temperature=1., top_p=1., top_k=0, repetition_penalty=1.,
                                           max_new_tokens=a.max_tokens, eos_token_id=stops, pad_token_id=tok.pad_token_id,
                                           use_cache=True)
            live, rewards, old = [], [], []
            for response in generated[:, ids_tensor.size(1):].tolist():
                for j, token in enumerate(response):
                    if token in stops:
                        response = response[:j+1]
                        break
                row = dict(positive, response_ids=response)
                value, method = prediction(tok.decode(response, skip_special_tokens=True).replace(r'\,', ' '))
                capped = response[-1] not in stops
                reward = float(value == gold(row['gold']) and method == 'boxed' and not capped) - .1*float(capped)
                live.append(row)
                rewards.append(reward)
                with torch.no_grad():
                    old.append(sequence_lp(model, row).detach())
                with (out / 'own_rollouts.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step+1, example_id=example, response_ids=response, reward=reward))+'\n')
            del generated, ids_tensor
            advantage = advantages(torch.tensor(rewards, device='cuda'))
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            for j, row in enumerate(live):
                policy = policy_loss(sequence_lp(model, row), old[j], advantage[j])
                master.backward(a.rl_weight * policy / a.group)
            live_kl = anchor(live[0], a.seed+700000+step)
            master.backward(a.anchor_weight * live_kl)
            reward_mean = sum(rewards)/len(rewards)
            del live, old, live_kl
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        pos_losses = []
        for j, row in enumerate([positive, other]):
            loss = anchor(row, a.seed+10000+step*2+j)
            master.backward(a.anchor_weight * loss / 2)
            pos_losses.append(float(loss.detach()))
            del loss
        ce = answer_ce(positive)
        master.backward(a.answer_weight * ce)
        ce_value = float(ce.detach())
        del ce
        anti_value, chosen, diagnostics = 0., [], None
        if weight:
            anti, chosen, diagnostics = negative_loss(negative, a.seed+20000+step)
            master.backward(weight * anti)
            anti_value = float(anti.detach())
            del anti
        norm = master.step()
        stats = dict(step=step+1, anti_weight=weight, anti_loss=anti_value, chosen_offsets=chosen,
                     negative_diagnostics=diagnostics, positive_kl=sum(pos_losses)/len(pos_losses), answer_ce=ce_value, teacher_reward=reward_mean,
                     grad_norm=norm, elapsed=time.time()-started, max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
        with (out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(stats)+'\n')
        manifest['completed_steps'] = step+1
        save()
        print('TRAIN', json.dumps(stats), flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    assert not hasattr(model, 'peft_config')
    with torch.no_grad():
        ref_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            if name in names:
                ref_params[name].copy_(parameter)
            else:
                assert torch.equal(parameter, ref_params[name]), 'Frozen parameter changed: '+name
        probe = torch.tensor([positives[eligible[0]]['prompt_ids']], device='cuda')
        before = model(input_ids=probe, use_cache=False).logits[:, -1].float()
        after = reference(input_ids=probe, use_cache=False).logits[:, -1].float()
        error = float((before-after).abs().max())
    assert error < .01
    model.save_pretrained(out / 'model', safe_serialization=True, max_shard_size='4GB')
    tok.save_pretrained(out / 'model')
    assert not (out / 'model/adapter_config.json').exists()
    manifest.update(complete=manifest['completed_steps'] == a.steps, plain_export_verified=True,
                    plain_reconstruction_logit_error=error, end=time.time())
    save()
    print('COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
