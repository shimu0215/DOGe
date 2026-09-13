"""Generate and freeze TinyLlama texts before a separate teacher-training process."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
os.environ.update(HF_HOME='/scratch/wzhao20/hf_cache',HF_HUB_CACHE='/scratch/wzhao20/hf_cache/hub',HF_DATASETS_CACHE='/scratch/wzhao20/DOGe-official/data/gsm8k_hf_cache_20260908',XDG_CACHE_HOME='/scratch/wzhao20/.cache',TORCH_HOME='/scratch/wzhao20/.cache/torch')
import torch
from huggingface_hub import HfApi,snapshot_download
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/opd_update_20260911'
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
base=OUT/'static_entropy4_mix14b_23371_context';m=json.loads((base/'manifest.json').read_text());assert m['complete']
ps=Path(m['prompts']);assert hashlib.sha256(ps.read_bytes()).hexdigest()==m['prompt_sha256'];prompts=[json.loads(s) for s in ps.read_text().splitlines()]
rows=[json.loads(s) for s in (base/'rollouts.jsonl').read_text().splitlines()];neg={r['example_id']:r for r in rows if r['source']=='student'};ids=sorted(neg);assert len(ids)==384
old14=set(m['replacement_ids']);rng=random.Random(20260913);selected=sorted(rng.sample([i for i in ids[:-64] if i not in old14],80)+rng.sample([i for i in ids[-64:] if i not in old14],16));assert not set(selected)&old14
repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0';revision=HfApi().model_info(repo).sha
info=dict(complete=False,repo=repo,revision=revision,model_card='https://huggingface.co/'+repo,start=time.time(),selected_ids=selected,teacher_training=False)
(out/'generation_manifest.json').write_text(json.dumps(info,indent=2))
path=snapshot_download(repo,revision=revision,local_dir=str(ROOT/'results/generalization_20260913/tinyllama-1.1b-chat-v1.0'),cache_dir='/scratch/wzhao20/hf_cache/hub',allow_patterns=['*.json','*.txt','*.model','*.jinja','*.safetensors'],max_workers=4)
tok=AutoTokenizer.from_pretrained(path);tok.padding_side='left';tok.pad_token=tok.eos_token
tt=AutoTokenizer.from_pretrained(m['teacher']);model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval();model.requires_grad_(False)
responses={};torch.manual_seed(20260913)
for start in range(0,len(selected),8):
 batch=selected[start:start+8];texts=[]
 for i in batch:
  ex=prompts[neg[i]['dataset_index']];assert ex['source_id']==neg[i]['source_id'] and ex['source_id']<1000
  texts.append(tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':ex['instruction']}],tokenize=False,add_generation_prompt=True))
 encoded=tok(texts,padding=True,return_tensors='pt').to('cuda')
 with torch.no_grad():generated=model.generate(**encoded,max_new_tokens=384,do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
 for i,tokens in zip(batch,generated[:,encoded.input_ids.shape[1]:].tolist()):
  ended=tok.eos_token_id in tokens
  if ended:tokens=tokens[:tokens.index(tok.eos_token_id)]
  text=tok.decode(tokens,skip_special_tokens=True);response=tt.encode(text,add_special_tokens=False)
  if ended:response.append(tt.eos_token_id)
  assert response,'Empty generator response; retain failed prep'
  responses[i]=dict(ids=response[:384],text=tt.decode(response[:384],skip_special_tokens=True),hit_cap=not ended or len(response)>384)
  with (out/'generated_texts.jsonl').open('a') as f:f.write(json.dumps(dict(example_id=i,source_id=neg[i]['source_id'],text=text,ended=ended))+'\n')
 info['generated']=len(responses);(out/'generation_manifest.json').write_text(json.dumps(info,indent=2));print('GENERATED',len(responses),flush=True)
for row in rows:
 if row['source']=='student' and row['example_id'] in responses:
  v=responses[row['example_id']];row=dict(row,response_ids=v['ids'],text=v['text'],hit_cap=v['hit_cap'],negative_generator=repo+'@'+revision+'; fixed text only')
 with (out/'rollouts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
assert len(responses)==96
info.update(complete=True,end=time.time(),path=path,sampling=dict(temperature=1.,top_p=1.,top_k=0,max_new_tokens=384));(out/'generation_manifest.json').write_text(json.dumps(info,indent=2))
m['raw_replacement_ids']=[i for i in m['raw_replacement_ids'] if i not in selected]
m.update(output=str(out),source_context=str(base),source_context_sha256=hashlib.sha256((base/'rollouts.jsonl').read_bytes()).hexdigest(),student=m['student']+[repo+' fixed texts'],complete=True,offline_sources_only=True,generator_models_loaded=True,generator_use='TinyLlama loaded only to produce fixed texts in this preparation process, which exits before teacher training. No generator logits/gradients/rewards saved.',additional_replacement_ids=selected,additional_train_count=80,additional_validation_count=16,source_counts={'qwen14b':192,'qwen0p5b':96,'tinyllama':96},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),end=time.time())
(out/'manifest.json').write_text(json.dumps(m,indent=2));print('COMPLETE',json.dumps(m),flush=True)
