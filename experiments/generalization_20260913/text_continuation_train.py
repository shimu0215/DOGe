"""Cross-tokenizer sampled continuation KD at online student prefixes.

This is text-continuation imitation, not the original tokenwise MiniLLM KL.
Only the student is optimized; teacher is frozen throughout.
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/opd_update_20260911'))
from direct_master import MasterAdam

def main():
 p=argparse.ArgumentParser()
 for k in ['student','teacher','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--steps',type=int,default=80);p.add_argument('--save-every',type=int,default=40)
 p.add_argument('--lr',type=float,default=3e-6);p.add_argument('--batch',type=int,default=4)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--teacher-tokens',type=int,default=128)
 a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
 assert torch.cuda.device_count()==1
 torch.set_num_threads(4);torch.manual_seed(a.seed);rng=random.Random(a.seed)
 st=AutoTokenizer.from_pretrained(a.student);tt=AutoTokenizer.from_pretrained(a.teacher)
 assert st.get_vocab()!=tt.get_vocab(), 'This experiment is specifically cross-tokenizer'
 s=AutoModelForCausalLM.from_pretrained(a.student,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
 t=AutoModelForCausalLM.from_pretrained(a.teacher,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
 t.requires_grad_(False);s.requires_grad_(True)
 master=MasterAdam(s.parameters(),lr=a.lr,scale=1.)
 data=load_dataset('openai/gsm8k','main',split='train');ids=list(range(1000));rng.shuffle(ids)
 system='Please reason step by step, and put your final answer within \\boxed{{}}.'
 def head(tok,q):return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':q}],tokenize=True,add_generation_prompt=True)
 def stops(model,tok):
  v=model.generation_config.eos_token_id;return sorted(set((v if isinstance(v,list) else [v])+[tok.eos_token_id])-{None})
 ss,ts=stops(s,st),stops(t,tt)
 started=time.time();record=dict(vars(a),complete=False,steps_done=0,student_only_optimization=True,teacher_frozen=True,teacher_parameter_feedback=False,
  scope='Online student-prefix text-continuation imitation with teacher sampled continuations; NOT tokenwise KL/MiniLLM, no cross-vocab logits comparison.',
  train_ids=ids,train_questions_sha256=hashlib.sha256(json.dumps([data[i]['question'] for i in ids]).encode()).hexdigest(),
  sampling=dict(student_temperature=1.,teacher_temperature=1.,top_p=1.,top_k=0,prefix_lengths=[32,64,96],student_rollout_max=128),
  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoints={})
 def save_record():(out/'manifest.json').write_text(json.dumps(record,indent=2))
 save_record()
 for step in range(a.steps):
  examples=[];s.eval();s.gradient_checkpointing_disable()
  for j in range(a.batch):
   idx=ids[(step*a.batch+j)%len(ids)];q=data[idx]['question'];sh=head(st,q);x=torch.tensor([sh],device='cuda')
   with torch.no_grad():generated=s.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=128,do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,eos_token_id=ss,pad_token_id=st.pad_token_id,use_cache=True)[0,len(sh):].tolist()
   for n,v in enumerate(generated):
    if v in ss:generated=generated[:n];break
   cap=rng.choice([32,64,96]);prefix=st.decode(generated[:cap],skip_special_tokens=True)
   # Never condition after an already-completed final answer.
   if '\\boxed' in prefix:prefix=prefix.split('\\boxed',1)[0]
   th=head(tt,q)+tt.encode(prefix,add_special_tokens=False);x=torch.tensor([th],device='cuda')
   with torch.no_grad():new=t.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=a.teacher_tokens,do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,eos_token_id=ts,pad_token_id=tt.pad_token_id,use_cache=True)[0,len(th):].tolist()
   ended=bool(new and new[-1] in ts);continuation=tt.decode(new,skip_special_tokens=True)
   encoded=st(prefix+continuation,add_special_tokens=False,return_offsets_mapping=True)
   tail=list(encoded['input_ids']);offsets=encoded['offset_mapping'];labels=[token if end>len(prefix) else -100 for token,(begin,end) in zip(tail,offsets)]
   if ended:tail.append(st.eos_token_id);labels.append(st.eos_token_id)
   assert any(v!=-100 for v in labels), 'Empty continuation target; retain failure for audit'
   boundary_overlap=sum(begin<len(prefix)<end for begin,end in offsets)
   assert boundary_overlap<=1
   assert len(sh)+len(tail)<=1024
   item=dict(id=idx,question=q,prefix=prefix,continuation=continuation,teacher_ended=ended,boundary_overlap=boundary_overlap,
     input_ids=sh+tail,labels=[-100]*len(sh)+labels)
   examples.append(item)
   with (out/'online_examples.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,**item))+'\n')
  master.zero_grad();s.train();s.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  losses=[]
  before=master.weights[0].detach().clone()
  for item in examples:
   x=torch.tensor([item['input_ids']],device='cuda');labels=torch.tensor([item['labels']],device='cuda')
   loss=s(input_ids=x,attention_mask=torch.ones_like(x),labels=labels,use_cache=False).loss
   losses.append(float(loss.detach()));master.backward(loss/a.batch)
  norm=master.step();delta=float((master.weights[0].detach()-before).square().mean().sqrt());del before
  assert delta>0 and all(torch.isfinite(torch.tensor(losses)))
  stats=dict(step=step+1,loss=sum(losses)/len(losses),grad_norm=norm,first_parameter_master_delta_rms=delta,elapsed=time.time()-started,max_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
  record['steps_done']=step+1
  if (step+1)%a.save_every==0 or step+1==a.steps:
   dest=out/('checkpoint-'+str(step+1));s.save_pretrained(dest,safe_serialization=True,max_shard_size='4GB');st.save_pretrained(dest);record['checkpoints'][str(step+1)]=str(dest)
  save_record();print(json.dumps(stats),flush=True)
 record.update(complete=True,end=time.time());save_record()
if __name__=='__main__':main()
