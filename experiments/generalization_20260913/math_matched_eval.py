"""Matched native-chat MATH calibration only; no OPD or defense claims."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=int,default=0);a=p.parse_args()
root=Path(__file__).resolve().parents[2];source=root/'AgentDistill/data_processor/math_dataset/test/math_500_20250414.json'
rows=json.loads(source.read_text())['examples'];assert len(rows)==500
rows=rows[a.start:a.start+100];assert len({r['question'] for r in rows})==100
out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(4);torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained(a.model);tok.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
stops=[151643,151645];assert tok.eos_token_id in stops and tok.pad_token_id in stops
results=[];started=time.time()
for begin in range(0,100,4):
 batch=rows[begin:begin+4];prompts=[tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':r['question']}],tokenize=False,add_generation_prompt=True) for r in batch]
 inputs=tok(prompts,return_tensors='pt',padding=True).to('cuda')
 with torch.no_grad():seq=model.generate(**inputs,max_new_tokens=1024,do_sample=False,eos_token_id=stops,pad_token_id=tok.pad_token_id,repetition_penalty=1.,use_cache=True)
 for r,prompt,ids in zip(batch,prompts,seq[:,inputs['input_ids'].shape[1]:].tolist()):
  ended=False
  for j,t in enumerate(ids):
   if t in stops:ids=ids[:j+1];ended=True;break
  results.append(dict(**r,prompt=prompt,prediction=tok.decode(ids,skip_special_tokens=True),generated_tokens=len(ids),hit_cap=not ended))
 (out/'predictions.json').write_text(json.dumps(dict(model=a.model,dataset_source=str(source),dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),split='test',indices=[a.start,a.start+100],scope='Validation0:100 or fixedtest100:200; explicit original Qwen stop set',generation=dict(do_sample=False,max_new_tokens=1024,repetition_penalty=1.,dtype='float16',eos_token_id=stops),content=results,complete=len(results)==100,elapsed=time.time()-started),indent=2))
 print('generated',len(results),flush=True)
