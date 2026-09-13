"""Native teacher/student headroom on official SVAMP; no training on challenge set."""
import argparse,hashlib,json,sys,time,urllib.request,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913';w=c.Worker(a.job,'svamp_smol_headroom_'+a.job,minimum=300)
try:
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 data=c.OUT/('svamp_headroom_'+a.job+'_data')
 payload=(data/'SVAMP.json').read_bytes();manifest=c.read(data/'manifest.json');assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
 rows=json.loads(payload);assert len(rows)==1000
 w.state['source_manifest']=manifest;w.state['original_teacher_reference']='svamp_headroom_fixed_22481_original_teacher.json';w.save()
 rows=rows[:100];assert len({r['ID'] for r in rows})==100
 w.state['scope']='SVAMP first100 calibration headroom, raw Smol1.7 native chat and model-native EOS versus existing Qwen teacher; compare same IDs/questions/gold, rendered tokenizer prompts differ; no OPD baseline or defense. Challenge set never used for training. Any later OPD requires a separate appropriate training corpus.';w.save()
 docs={'original_teacher':c.read(c.OUT/'svamp_headroom_fixed_22481_original_teacher.json')['content']}
 w.state['results']={'original_teacher':dict(n=100,correct=sum(r['correct'] for r in docs['original_teacher']))};w.save()
 for label,path in [('raw_smol17_student',c.OUT/'smollm2-1.7b-instruct')]:
  w.state['phase']=label;w.save();tok=AutoTokenizer.from_pretrained(path);tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval();result=[]
  stops=model.generation_config.eos_token_id or tok.eos_token_id
  assert stops is not None and tok.pad_token_id is not None
  for start in range(0,100,4):
   batch=rows[start:start+4];prompts=[tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':r['Body'].strip()+' '+r['Question'].strip()}],tokenize=False,add_generation_prompt=True) for r in batch]
   x=tok(prompts,padding=True,return_tensors='pt').to('cuda')
   with torch.no_grad():seq=model.generate(**x,max_new_tokens=512,do_sample=False,eos_token_id=stops,pad_token_id=tok.pad_token_id,repetition_penalty=1.)
   for row,prompt,tokens in zip(batch,prompts,seq[:,x.input_ids.shape[1]:].tolist()):
    text=tok.decode(tokens,skip_special_tokens=True);correct=int(c.prediction(text.replace(chr(92)+',',' '))[0]==c.gold('#### '+str(row['Answer'])))
    result.append(dict(id=row['ID'],prompt=prompt,prediction=text,ground_truth='#### '+str(row['Answer']),correct=correct))
   c.write(c.OUT/(w.tag+'_'+label+'.json'),dict(complete=len(result)==100,model=str(path),generation=dict(do_sample=False,max_new_tokens=512,dtype='float16',eos_token_id=stops,repetition_penalty=1.),content=result))
  docs[label]=result;w.state.setdefault('results',{})[label]=dict(n=100,correct=sum(r['correct'] for r in result));w.save();del model,x,seq;gc.collect();torch.cuda.empty_cache()
 assert [(r['id'],r['ground_truth']) for r in docs['original_teacher']]==[(r['id'],r['ground_truth']) for r in docs['raw_smol17_student']]
 w.state['teacher_headroom']=w.state['results']['original_teacher']['correct']-w.state['results']['raw_smol17_student']['correct'];w.finish()
except Exception as e:w.fail(e);raise
