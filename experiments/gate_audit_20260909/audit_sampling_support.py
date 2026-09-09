"""Measure actual student action-support restrictions after decoding processors."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,LogitsProcessor,LogitsProcessorList

root=Path('results');meta=json.loads((root/'context96/manifest.json').read_text())
rows=[json.loads(x) for x in (root/'context96/rollouts.jsonl').read_text().splitlines() if json.loads(x)['source']=='student'][:8]
tok=AutoTokenizer.from_pretrained(meta['student']);tok.padding_side='left';tok.pad_token_id=tok.eos_token_id
model=AutoModelForCausalLM.from_pretrained(meta['student'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
counts=[]
class Support(LogitsProcessor):
    def __init__(self,start):self.start=start
    def __call__(self,ids,scores):
        active=torch.ones(len(ids),dtype=torch.bool,device=ids.device)
        if ids.size(1)>self.start:active=ids[:,-1].ne(tok.eos_token_id)
        counts.extend(scores.isfinite().sum(-1)[active].cpu().tolist())
        return scores
original=model._get_logits_processor
for start in range(0,len(rows),2):
    prompts=[r['prompt_ids'] for r in rows[start:start+2]];width=max(map(len,prompts))
    ids=torch.tensor([[tok.pad_token_id]*(width-len(x))+x for x in prompts],device='cuda')
    tracker=Support(width)
    model._get_logits_processor=lambda *a,**kw:LogitsProcessorList(list(original(*a,**kw))+[tracker])
    torch.manual_seed(42+start)
    with torch.inference_mode():
        model.generate(input_ids=ids,attention_mask=ids.ne(tok.pad_token_id),do_sample=True,
            temperature=.7,top_p=.8,top_k=0,repetition_penalty=1.1,max_new_tokens=384,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_model_defaults=False)
    print('support sampled',start+len(prompts),flush=True)
result={'n_sequences':len(rows),'n_positions':len(counts),'singleton_fraction':sum(x==1 for x in counts)/len(counts),
    'at_most_five_fraction':sum(x<=5 for x in counts)/len(counts),'median_candidate_count':sorted(counts)[len(counts)//2],
    'actual_policy':{'temperature':.7,'top_p':.8,'top_k':0,'repetition_penalty':1.1},
    'mask':'Matches existing MiniLLM pad=eos and token-value attention mask',
    'scope':'Eight diagnostic prompts; descriptive support audit, not a training-wide estimate'}
(root/'sampling_support.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
