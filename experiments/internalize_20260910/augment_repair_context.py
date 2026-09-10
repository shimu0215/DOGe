"""Add matched trajectories from unused TRAIN prompts, without touching GSM test."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

p=argparse.ArgumentParser()
for name in ['teacher','student','old-context','prompts','output']:p.add_argument('--'+name,required=True)
p.add_argument('--n',type=int,default=320);p.add_argument('--tokens',type=int,default=512)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
old=[json.loads(x) for x in Path(a.old_context).read_text().splitlines()]
all_rows=[json.loads(x) for x in Path(a.prompts).read_text().splitlines()]
used={r['dataset_index'] for r in old}
picked=random.Random(20260910).sample(sorted(set(range(len(all_rows)))-used),a.n)
offset=1+max(r['example_id'] for r in old)
manifest=dict(vars(a),start=time.time(),new_prompt_indices=picked,new_example_offset=offset,
    disjoint_from_original_context_prompts=True,split='GSM8K TRAIN prompts already audited against official train',
    prompts_sha256=hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest(),
    old_context_sha256=hashlib.sha256(Path(a.old_context).read_bytes()).hexdigest(),
    source_dtypes={'student':'bfloat16','teacher':'float16','teacher_greedy':'float16'},
    sampling={'temperature':1.,'top_p':1.,'top_k':0,'repetition_penalty':1.})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
destination=out/'rollouts.jsonl';destination.write_text(Path(a.old_context).read_text())
tok=AutoTokenizer.from_pretrained(a.teacher);tok.padding_side='left'
assert tok.get_vocab()==AutoTokenizer.from_pretrained(a.student).get_vocab()
with torch.inference_mode():
    for path,dtype,sources in [(a.student,torch.bfloat16,[('student',True,1)]),
        (a.teacher,torch.float16,[('teacher',True,0),('teacher_greedy',False,2)])]:
        model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=dtype,
            low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id,tok.pad_token_id}
        stops=sorted(stops-{None})
        for source,sampling,seed_offset in sources:
            for start in range(0,a.n,8):
                batch=picked[start:start+8];prompts=[all_rows[i]['prompt'] for i in batch]
                enc=tok(prompts,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
                torch.manual_seed(20260910+10000*seed_offset+start)
                gen=model.generate(**enc,do_sample=sampling,temperature=1.,top_p=1.,top_k=0,
                    repetition_penalty=1.,max_new_tokens=a.tokens,eos_token_id=stops,pad_token_id=tok.pad_token_id)
                for j,(index,response) in enumerate(zip(batch,gen[:,enc.input_ids.size(1):].tolist())):
                    for k,token in enumerate(response):
                        if token in stops:response=response[:k+1];break
                    row={'source':source,'example_id':offset+start+j,'dataset_index':index,
                        'source_id':all_rows[index].get('source_id'),
                        'prompt_ids':tok.encode(prompts[j],add_special_tokens=False),'response_ids':response,
                        'text':tok.decode(response,skip_special_tokens=True),
                        'hit_cap':len(response)==a.tokens and response[-1] not in stops}
                    with destination.open('a') as f:f.write(json.dumps(row)+'\n')
                print('GENERATE',source,min(start+8,a.n),a.n,flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
merged=[json.loads(x) for x in destination.read_text().splitlines()]
assert len(merged)==len(old)+3*a.n
assert len({(r['example_id'],r['source']) for r in merged})==len(merged)
manifest.update(complete=True,end=time.time(),total_rows=len(merged),
    rollout_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',out,flush=True)
