"""Native teacher/student headroom on official SVAMP; no training on challenge set."""
import argparse,hashlib,json,sys,time,urllib.request,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913';w=c.Worker(a.job,'svamp_headroom_'+a.job,minimum=600)
try:
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 req=urllib.request.Request('https://api.github.com/repos/arkilpatel/SVAMP/commits/main',headers={'User-Agent':'research-audit'})
 revision=json.load(urllib.request.urlopen(req,timeout=60))['sha'];url='https://raw.githubusercontent.com/arkilpatel/SVAMP/'+revision+'/SVAMP.json';payload=urllib.request.urlopen(url,timeout=60).read();rows=json.loads(payload);assert len(rows)==1000
 data=c.OUT/(w.tag+'_data');data.mkdir(exist_ok=False);(data/'SVAMP.json').write_bytes(payload)
 c.write(data/'manifest.json',dict(repo='https://github.com/arkilpatel/SVAMP',revision=revision,url=url,sha256=hashlib.sha256(payload).hexdigest(),n=1000,scope='Official challenge set, evaluation only; no training split invented'))
 rows=rows[:100];assert len({r['ID'] for r in rows})==100
 w.state['scope']='SVAMP first100 calibration headroom only; no OPD baseline or defense. Challenge set never used for training. Any later OPD requires a separate appropriate training corpus.';w.save()
 docs={}
 for label,path in [('original_teacher',c.ORIGINAL),('gsm_sft_student',ROOT/'results/baseline_20260911/short_sft/checkpoint-49')]:
  w.state['phase']=label;w.save();tok=AutoTokenizer.from_pretrained(path);tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval();result=[]
  for start in range(0,100,4):
   batch=rows[start:start+4];prompts=[tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':r['Body'].strip()+' '+r['Question'].strip()}],tokenize=False,add_generation_prompt=True) for r in batch]
   x=tok(prompts,padding=True,return_tensors='pt').to('cuda')
   with torch.no_grad():seq=model.generate(**x,max_new_tokens=512,do_sample=False,eos_token_id=[151643,151645],pad_token_id=tok.pad_token_id,repetition_penalty=1.)
   for row,prompt,tokens in zip(batch,prompts,seq[:,x.input_ids.shape[1]:].tolist()):
    text=tok.decode(tokens,skip_special_tokens=True);correct=int(c.prediction(text.replace(chr(92)+',',' '))[0]==c.gold(str(row['Answer'])))
    result.append(dict(id=row['ID'],prompt=prompt,prediction=text,ground_truth=str(row['Answer']),correct=correct))
   c.write(c.OUT/(w.tag+'_'+label+'.json'),dict(complete=len(result)==100,model=str(path),generation=dict(do_sample=False,max_new_tokens=512,dtype='float16',eos_token_id=[151643,151645],repetition_penalty=1.),content=result))
  docs[label]=result;w.state.setdefault('results',{})[label]=dict(n=100,correct=sum(r['correct'] for r in result));w.save();del model,x,seq;gc.collect();torch.cuda.empty_cache()
 assert [(r['id'],r['prompt'],r['ground_truth']) for r in docs['original_teacher']]==[(r['id'],r['prompt'],r['ground_truth']) for r in docs['gsm_sft_student']]
 w.state['teacher_headroom']=w.state['results']['original_teacher']['correct']-w.state['results']['gsm_sft_student']['correct'];w.finish()
except Exception as e:w.fail(e);raise
