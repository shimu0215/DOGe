"""Same audited native Qwen SVAMP generation and numeric scoring."""
import argparse,json,sys,hashlib,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=int,default=0);a=p.parse_args()
source=ROOT/'results/generalization_20260913/svamp_headroom_22481_data/SVAMP.json';rows=c.read(source)[a.start:a.start+100];assert len(rows)==100
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained(a.model);tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval();assert tok.eos_token_id in [151643,151645]
results=[]
for begin in range(0,100,4):
 batch=rows[begin:begin+4];prompts=[tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':r['Body'].strip()+' '+r['Question'].strip()}],tokenize=False,add_generation_prompt=True) for r in batch];x=tok(prompts,padding=True,return_tensors='pt').to('cuda')
 with torch.no_grad():seq=model.generate(**x,max_new_tokens=512,do_sample=False,eos_token_id=[151643,151645],pad_token_id=tok.pad_token_id,repetition_penalty=1.)
 for r,prompt,ids in zip(batch,prompts,seq[:,x.input_ids.shape[1]:].tolist()):
  text=tok.decode(ids,skip_special_tokens=True);gold='#### '+str(r['Answer']);correct=int(c.prediction(text.replace(chr(92)+',',' '))[0]==c.gold(gold));results.append(dict(id=r['ID'],prompt=prompt,prediction=text,ground_truth=gold,correct=correct))
 c.write(out/'gsm8k-results.json',dict(complete=len(results)==100,model=a.model,dataset_source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),indices=[a.start,a.start+100],generation=dict(do_sample=False,max_new_tokens=512,dtype='float16',eos_token_id=[151643,151645],repetition_penalty=1.),content=results));print('Generated',len(results),flush=True)
