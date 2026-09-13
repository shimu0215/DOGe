"""Raw pretrained-model diagnostic using a completion prompt, no chat-template assumption."""
import argparse,json,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);assert not out.exists();out.mkdir()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold
rows=load_dataset('openai/gsm8k','main',split='train').select(range(7000,7100))
t=AutoTokenizer.from_pretrained(a.model);t.padding_side='left'
if t.pad_token_id is None:t.pad_token=t.eos_token
m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
result=[]
for start in range(0,100,8):
 batch=rows.select(range(start,min(start+8,100)))
 prompts=['Question: '+q+'\nAnswer: Let\'s think step by step.\n' for q in batch['question']]
 encoded=t(prompts,padding=True,return_tensors='pt').to('cuda')
 with torch.no_grad():outids=m.generate(**encoded,max_new_tokens=512,do_sample=False,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id)
 texts=t.batch_decode(outids[:,encoded.input_ids.shape[1]:],skip_special_tokens=True)
 for j,(q,ans,text,prompt) in enumerate(zip(batch['question'],batch['answer'],texts,prompts)):
  correct=int(prediction(text.replace(chr(92)+',',' '))[0]==gold(ans))
  result.append(dict(id=7000+start+j,question=q,prompt=prompt,ground_truth=ans,prediction=text,correct=correct))
 print('evaluated',len(result),flush=True)
(out/'results.json').write_text(json.dumps(dict(model=a.model,split='train7000:7100',n=100,correct=sum(x['correct'] for x in result),protocol='Raw base zero-shot completion, greedy512; different native formatting from instruct teacher, not a maximum-capability estimate',content=result),indent=2))
