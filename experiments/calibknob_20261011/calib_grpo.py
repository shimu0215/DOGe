"""复现 Zhan et al. 2026 (arXiv 2604.18963) "Distillation Traps and Guards" 的 undistillable 校准 (Alg. 1).

J(pi) = E_{s~pi}[ R_task(s) - beta*KL(pi||pi_T) + eta*KL(pi||pi_C) ]
  R_calib(s) = sg[log pi_theta(s)] - log pi_C(s)   (序列级, 只在 completion token 上求和)
  A_i = GroupNorm(R_task)_i + eta * GroupNorm(R_calib)_i   (两项各自组内归一化)
原文给出: eta=+1, LoRA r=128 alpha=128, lr 2e-5, 每题 4 rollouts, 基于 TRL GRPO.
原文未给出(此处自定): beta=0.04(TRL GRPO 旧默认), k3 KL 估计, 每步 8 题, 单 PPO epoch(ratio=1, 无需 clip),
  token 级平均损失, T=1.0/top-p 1/top-k 0, 512 新 token, LoRA 作用于全部线性层.
任务: CSQA 训练集(ChatML + \\boxed{字母}, 与 mcq_eval.py 同 prompt/抽取), pi_C = 未 SFT 的 Qwen2.5-0.5B-Instruct.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

p = argparse.ArgumentParser()
p.add_argument('--teacher', required=True); p.add_argument('--proxy', required=True); p.add_argument('--output', required=True)
p.add_argument('--mcq-dir', required=True, help='含 mcq_data.py / mcq_eval.py 的目录')
p.add_argument('--dataset', default='csqa'); p.add_argument('--steps', type=int, default=120); p.add_argument('--max-minutes', type=float, default=70)
p.add_argument('--prompts-per-step', type=int, default=8); p.add_argument('--group', type=int, default=4)
p.add_argument('--eta', type=float, default=1.0); p.add_argument('--beta', type=float, default=0.04)
p.add_argument('--lr', type=float, default=2e-5); p.add_argument('--rank', type=int, default=128); p.add_argument('--alpha', type=int, default=128)
p.add_argument('--max-new', type=int, default=512); p.add_argument('--micro', type=int, default=4); p.add_argument('--seed', type=int, default=0)
p.add_argument('--save-every', type=int, default=0, help='每 N 步存 LoRA adapter 到 ckpt/step_N(全部保留), 并覆盖 resume/ 下的优化器与指针')
p.add_argument('--resume', action='store_true', help='从 output/resume 续训(output 目录已存在)')
p.add_argument('--no-merge', action='store_true', help='结束时不合并导出完整模型(用 merge_ckpt.py 另行合并)')
a = p.parse_args()
sys.path.insert(0, a.mcq_dir)
from mcq_data import load_mcq, SYS
from mcq_eval import extract

random.seed(a.seed); torch.manual_seed(a.seed)
out = Path(a.output); out.mkdir(parents=True, exist_ok=a.resume)
(out / ('args_resume_%d.json' % int(time.time()) if a.resume else 'args.json')).write_text(json.dumps(vars(a), indent=2))
train, _ = load_mcq(a.dataset); random.shuffle(train)
tok = AutoTokenizer.from_pretrained(a.teacher); tok.padding_side = 'left'
stops = [151643, 151645]; pad = tok.pad_token_id
base = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); base.enable_input_require_grads()
model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, task_type='CAUSAL_LM',
                                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
model.print_trainable_parameters()
proxy = AutoModelForCausalLM.from_pretrained(a.proxy, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)

def seq_logp(m, ids, attn, cmask):
    """逐 token log p(completion token), 形状 [B, L-1], 非 completion 位置为 0.
    teacher 词表(152064)比 proxy(151936)多 128 个无文本的保留 id; 原文 R_calib 按文本重新分词打分,
    这些 id 解码为空串 => 对 proxy 视为不存在: 输入替换为 pad, 且不计入 logp."""
    V = m.get_input_embeddings().weight.shape[0]
    bad = ids >= V
    if bad.any():
        ids = ids.masked_fill(bad, pad); cmask = cmask * (~bad).float()
    logits = m(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, :-1].float()
    tgt = ids[:, 1:].clamp(max=logits.shape[-1] - 1)  # proxy 词表 151936 < teacher 152064, 防越界(仅可能是 pad 位)
    lp = torch.gather(F.log_softmax(logits, -1), 2, tgt[..., None]).squeeze(-1)
    return lp * cmask[:, 1:]

def gnorm(x):  # 组内 z-score, x: [P, G]
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-4)

start = 1; ptr = 0
if a.resume:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    st = json.loads((out / 'resume' / 'state.json').read_text())
    set_peft_model_state_dict(model, load_file(str(out / 'resume' / 'adapter' / 'adapter_model.safetensors')))
    opt.load_state_dict(torch.load(out / 'resume' / 'opt.pt', map_location='cuda'))
    start, ptr = st['step'] + 1, st['ptr']
    random.setstate(tuple(st['py_rng'][:1]) + (tuple(st['py_rng'][1]),) + tuple(st['py_rng'][2:])); torch.manual_seed(a.seed + st['step'])
    print(f'RESUMED from step {st["step"]} ptr {ptr}', flush=True)

def save_ckpt(step):
    model.save_pretrained(out / 'ckpt' / f'step_{step}')
    tmp = out / 'resume_tmp'; tmp.mkdir(exist_ok=True)
    model.save_pretrained(tmp / 'adapter'); torch.save(opt.state_dict(), tmp / 'opt.pt')
    (tmp / 'state.json').write_text(json.dumps(dict(step=step, ptr=ptr, py_rng=random.getstate())))
    old = out / 'resume_old'
    if (out / 'resume').exists(): (out / 'resume').rename(old)
    tmp.rename(out / 'resume')
    if old.exists(): import shutil; shutil.rmtree(old)
    print(f'CKPT step {step}', flush=True)

log = open(out / 'train_log.jsonl', 'a'); t0 = time.time()
for step in range(start, a.steps + 1):
    rows = train[ptr:ptr + a.prompts_per_step]; ptr += a.prompts_per_step
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': r['question']}], tokenize=False, add_generation_prompt=True) for r in rows]
    enc = tok(prompts, return_tensors='pt', padding=True).to('cuda'); P = enc['input_ids'].shape[1]
    model.eval()
    with torch.no_grad():
        seq = model.generate(input_ids=enc['input_ids'].repeat_interleave(a.group, 0), attention_mask=enc['attention_mask'].repeat_interleave(a.group, 0),
                             do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new, eos_token_id=stops, pad_token_id=pad, use_cache=True)
    comp = seq[:, P:]; N = comp.shape[0]
    cm = torch.zeros_like(comp, dtype=torch.float)
    task, lens, caps, texts = [], [], [], []
    for i in range(N):
        ids = comp[i].tolist(); L = len(ids); ended = False
        for j, t in enumerate(ids):
            if t in stops: L = j + 1; ended = True; break
        cm[i, :L] = 1; lens.append(L); caps.append(not ended)
        text = tok.decode(ids[:L], skip_special_tokens=True); texts.append(text)
        r = rows[i // a.group]; task.append(1.0 if extract(text, r['labels']) == r['answer'] else 0.0)
    attn = torch.cat([enc['attention_mask'].repeat_interleave(a.group, 0), cm.long()], 1)
    cmask = torch.cat([torch.zeros_like(enc['attention_mask'].repeat_interleave(a.group, 0), dtype=torch.float), cm], 1)
    # 参考 teacher(关 LoRA)与 proxy 学生的逐 token logp, 以及当前策略的序列 logp(作 sg 奖励)
    with torch.no_grad():
        ref_lp, prox_lp, pol_seq = [], [], []
        for b in range(0, N, a.micro):
            s = slice(b, b + a.micro)
            with model.disable_adapter(): ref_lp.append(seq_logp(model, seq[s], attn[s], cmask[s]))
            prox_lp.append(seq_logp(proxy, seq[s], attn[s], cmask[s]))
        ref_lp = torch.cat(ref_lp); prox_lp = torch.cat(prox_lp)
    model.train(); opt.zero_grad(set_to_none=True)
    # 先算一遍当前策略 logp(no_grad)得到 R_calib 与优势, 再分块反传
    with torch.no_grad():
        model.eval(); pol_lp = torch.cat([seq_logp(model, seq[b:b + a.micro], attn[b:b + a.micro], cmask[b:b + a.micro]) for b in range(0, N, a.micro)]); model.train()
    r_task = torch.tensor(task, device='cuda').view(-1, a.group)
    r_cal = (pol_lp.sum(1) - prox_lp.sum(1)).view(-1, a.group)
    adv = (gnorm(r_task) + a.eta * gnorm(r_cal)).view(-1)
    ntok = cmask[:, 1:].sum(); kl_sum = 0.0; pg_sum = 0.0
    for b in range(0, N, a.micro):
        s = slice(b, b + a.micro); m = cmask[s, 1:]
        lp = seq_logp(model, seq[s], attn[s], cmask[s])
        d = (ref_lp[s] - lp) * m
        k3 = (torch.exp(d) - d - 1) * m
        pg = -(adv[s, None] * lp * m)
        loss = (pg.sum() + a.beta * k3.sum()) / ntok
        loss.backward(); kl_sum += k3.sum().item(); pg_sum += pg.sum().item()
    gn = torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
    opt.step()
    rec = dict(step=step, elapsed_min=round((time.time() - t0) / 60, 2), task_acc=float(r_task.mean()), r_calib=float(r_cal.mean()),
               r_calib_per_tok=float((r_cal.view(-1) / torch.tensor(lens, device='cuda')).mean()), kl_per_tok=kl_sum / ntok.item(),
               len=sum(lens) / N, cap=sum(caps) / N, reserved_tok=int(((comp >= 151936) & (cm > 0)).sum()), grad_norm=float(gn), corr_len_rcal=float(torch.corrcoef(torch.stack([torch.tensor(lens, device='cuda', dtype=torch.float), r_cal.view(-1)]))[0, 1]))
    if step % 10 == 1: rec['sample'] = texts[0][:600]
    print(json.dumps(rec, ensure_ascii=False), flush=True); log.write(json.dumps(rec, ensure_ascii=False) + '\n'); log.flush()
    if ptr + a.prompts_per_step > len(train): ptr = 0; print(f'EPOCH END at step {step}', flush=True)
    capped = (time.time() - t0) / 60 > a.max_minutes
    if a.save_every and (step % a.save_every == 0 or capped or step == a.steps): save_ckpt(step)
    if capped: print(f'time cap at step {step}', flush=True); break

if a.no_merge: print('DONE (no merge)', flush=True); sys.exit(0)
model.eval(); merged = model.merge_and_unload()
merged.save_pretrained(out / 'model', safe_serialization=True); tok.save_pretrained(out / 'model')
(out / 'manifest.json').write_text(json.dumps(dict(steps_done=step, elapsed_min=(time.time() - t0) / 60, model=str(out / 'model')), indent=2))
print('SAVED', out / 'model', flush=True)
