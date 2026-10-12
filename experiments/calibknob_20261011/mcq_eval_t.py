"""MCQ 评测(与 claude_20260914/mcq_eval.py 同 prompt/抽取), 增加 raw 模式 = T1.0/top-p 1/top-k 0(原文口径).
用法: --dataset csqa --model M --output DIR --mode greedy|raw [--start 0 --count 100 --max-tokens 512]"""
import argparse, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--model', required=True); p.add_argument('--output', required=True)
p.add_argument('--mode', choices=['greedy', 'raw'], required=True); p.add_argument('--start', type=int, default=0); p.add_argument('--count', type=int, default=100)
p.add_argument('--max-tokens', type=int, default=512); p.add_argument('--mcq-dir', default='/scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914'); a = p.parse_args()
sys.path.insert(0, a.mcq_dir)
from mcq_data import load_mcq, SYS
from mcq_eval import extract
torch.manual_seed(42)
_, test = load_mcq(a.dataset); rows = test[a.start:a.start + a.count]
out = Path(a.output); out.mkdir(exist_ok=False, parents=True); torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
stops = [151643, 151645]; results = []; t0 = time.time()
gk = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if a.mode == 'raw' else dict(do_sample=False)
for b in range(0, len(rows), 8):
    batch = rows[b:b + 8]
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': r['question']}], tokenize=False, add_generation_prompt=True) for r in batch]
    inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda'); torch.manual_seed(42 + b)
    with torch.no_grad():
        seq = model.generate(**inputs, max_new_tokens=a.max_tokens, eos_token_id=stops, pad_token_id=tok.pad_token_id, use_cache=True, **gk)
    for r, ids in zip(batch, seq[:, inputs['input_ids'].shape[1]:].tolist()):
        ended = False
        for j, t in enumerate(ids):
            if t in stops: ids = ids[:j + 1]; ended = True; break
        text = tok.decode(ids, skip_special_tokens=True); pred = extract(text, r['labels'])
        results.append(dict(id=r['id'], uid=r['uid'], question=r['question'], ground_truth=r['answer'], prediction=text, pred_letter=pred,
                            correct=(pred == r['answer']), boxed=('\\boxed' in text), generated_tokens=len(ids), hit_cap=not ended))
n = len(results); acc = sum(x['correct'] for x in results) / n
summ = dict(dataset=a.dataset, mode=a.mode, n=n, accuracy=acc, boxed_rate=sum(x['boxed'] for x in results) / n,
            cap_rate=sum(x['hit_cap'] for x in results) / n, mean_tokens=sum(x['generated_tokens'] for x in results) / n)
(out / 'predictions.json').write_text(json.dumps(dict(summ, model=a.model, indices=[a.start, a.start + n], content=results, elapsed=time.time() - t0), indent=2))
print(json.dumps(summ))
