"""Evaluate frozen-teacher changes on unseen-family text; this is not an OPD result."""
import argparse,gc,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
c.OUT=ROOT/'results/generalization_20260913';w=c.Worker(a.job,'foreign_prefix_'+a.job,minimum=3600)
try:
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from datasets import load_dataset
 sm=c.read(c.OUT/'download_smollm.json');assert sm['complete']
 foreign=Path(sm['models']['smollm2-360m-instruct']['path'])
 data=load_dataset('openai/gsm8k','main',split='train').select(range(7000,7064))
 manifest=c.read(c.DEFENSE.parent/'manifest.json')
 assert manifest['complete'] and manifest['plain_export_verified']
 assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 w.state.update(scope='Frozen 7B output-change diagnostics on independent SmolLM text, own text and whitespace-normalized own text. No teacher updates, no cross-tokenizer KL, no OPD efficacy/source-classification-accuracy claim.',n=64,validation='GSM train7000:7064',teacher_checkpoint=str(c.DEFENSE));w.save()
 system='Please reason step by step, and put your final answer within \\boxed{{}}.'
 def chat(t,q):return t.apply_chat_template([{'role':'system','content':system},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
 roles={}
 for label,path in [('foreign',foreign),('self',c.ORIGINAL)]:
  w.state['phase']='generate_'+label;w.save()
  t=AutoTokenizer.from_pretrained(path);t.padding_side='left'
  if t.pad_token_id is None:t.pad_token=t.eos_token
  m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
  rows=[]
  for start in range(0,64,8):
   batch=data.select(range(start,start+8));prompts=[chat(t,q) for q in batch['question']]
   encoded=t(prompts,padding=True,return_tensors='pt').to('cuda')
   with torch.no_grad():ids=m.generate(**encoded,do_sample=False,repetition_penalty=1.,max_new_tokens=512,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id)
   texts=t.batch_decode(ids[:,encoded.input_ids.shape[1]:],skip_special_tokens=True)
   for j,(q,ans,text) in enumerate(zip(batch['question'],batch['answer'],texts)):
    correct=int(c.prediction(text.replace(chr(92)+',',' '))[0]==c.gold(ans))
    rows.append(dict(id=7000+start+j,question=q,ground_truth=ans,response=text,correct=correct))
   w.state[label+'_generated']=len(rows);w.save()
  roles[label]=rows;del m,encoded,ids;gc.collect();torch.cuda.empty_cache()
 roles['self_compact']=[dict(r,response=' '.join(r['response'].split())) for r in roles['self']]
 path=c.OUT/(w.tag+'_rollouts.json');c.write(path,roles)
 w.state['generation_correct']={k:sum(r['correct'] for r in v) for k,v in roles.items()};w.state['rollouts']=str(path);w.save()
 t=AutoTokenizer.from_pretrained(c.ORIGINAL)
 assert t.get_vocab()==AutoTokenizer.from_pretrained(c.DEFENSE).get_vocab()
 reference=AutoModelForCausalLM.from_pretrained(c.ORIGINAL,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
 defense=AutoModelForCausalLM.from_pretrained(c.DEFENSE,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
 measurements=[]
 for role,rows in roles.items():
  w.state['phase']='score_'+role;w.save()
  for row in rows:
   prefix=t.encode(chat(t,row['question']),add_special_tokens=False)
   response=t.encode(row['response'],add_special_tokens=False)[:512]
   # Keep early and late boundaries out; identical rule for every source.
   available=list(range(32,max(32,len(response)-32)))
   if not available:continue
   offsets=sorted(random.Random(20000+row['id']).sample(available,min(32,len(available))))
   ids=torch.tensor([prefix+response[:max(offsets)+1]],device='cuda');positions=[len(prefix)-1+i for i in offsets]
   observed=torch.tensor([response[i] for i in offsets],device='cuda')[:,None]
   with torch.no_grad():
    def lp(model):
     hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,positions]
     return model.lm_head(hidden).float().log_softmax(-1)
    native=lp(reference);changed=lp(defense)
    result=dict(role=role,id=row['id'],correct=row['correct'],teacher_tokens=len(response),positions=len(offsets),forward_kl=float((native.exp()*(native-changed)).sum(-1).mean()),observed_logp_change=float((changed.gather(-1,observed)-native.gather(-1,observed)).mean()),argmax_flip_fraction=float((native.argmax(-1)!=changed.argmax(-1)).float().mean()))
   measurements.append(result)
  w.state['scored_rows']=len(measurements);w.save()
 def avg(rows,key):return sum(r[key] for r in rows)/len(rows) if rows else None
 summary={}
 for role in roles:
  selected=[r for r in measurements if r['role']==role]
  summary[role]=dict(n=len(selected),**{k:avg(selected,k) for k in ['forward_kl','observed_logp_change','argmax_flip_fraction']},by_correctness={str(v):dict(n=len([r for r in selected if r['correct']==v]),forward_kl=avg([r for r in selected if r['correct']==v],'forward_kl')) for v in [0,1]})
 c.write(c.OUT/(w.tag+'_measurements.json'),dict(summary=summary,rows=measurements))
 w.state['summary']=summary;w.state['interpretation_limit']='Distribution shifts can reflect errors, length, content or surface format; these summaries do not identify a causal source detector or establish suppression. Cross-student OPD still needs its own positive baseline.';w.finish()
except Exception as e:w.fail(e);raise
