"""Free generation with an ordinary saved model. No gate, proxy or hooks."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'gate_audit_20260909'))
from corrected_numeric_audit import prediction, gold, paired


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--examples',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=64)
    p.add_argument('--start',type=int,default=0);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--modes',nargs='+',choices=['greedy','sampling','raw','warm'],default=['greedy','sampling'])
    p.add_argument('--max-tokens',type=int,default=512);p.add_argument('--baseline')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads(Path(a.examples).read_text())['content'][a.start:a.start+a.limit]
    assert len(rows)==a.limit
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(a.model);tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    assert not hasattr(model,'peft_config'), 'Inference must be a plain standalone model'
    eos=model.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos]) | {tokenizer.pad_token_id,tokenizer.eos_token_id}
    stops.discard(None);stops=sorted(stops)
    summary={'model':a.model,'n':len(rows),'dtype':'float16','architecture':type(model).__name__,
             'inference_external_components':False,'start_time':time.time(),'modes':{},
             'example_file':a.examples,'example_sha256':hashlib.sha256(Path(a.examples).read_bytes()).hexdigest(),
             'indices':[r['id'] for r in rows]}
    for mode in a.modes:
        config={'do_sample':mode!='greedy','temperature':1. if mode in ['raw','warm'] else .7,
                'top_p':1. if mode=='raw' else .8,'top_k':0 if mode=='raw' else 20,
                'repetition_penalty':1. if mode=='raw' else 1.05}
        if mode=='greedy':config={'do_sample':False,'repetition_penalty':1.05}
        generated=[]
        for start in range(0,len(rows),a.batch):
            batch=rows[start:start+a.batch]
            enc=tokenizer([r['prompt'] for r in batch],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
            torch.manual_seed(42+start)
            gen=model.generate(**enc,**config,max_new_tokens=a.max_tokens,eos_token_id=stops,
                pad_token_id=tokenizer.pad_token_id,use_cache=True)
            for r,ids in zip(batch,gen[:,enc.input_ids.size(1):].tolist()):
                for i,t in enumerate(ids):
                    if t in stops:ids=ids[:i+1];break
                text=tokenizer.decode(ids,skip_special_tokens=True);value,method=prediction(text)
                item={'id':r['id'],'prompt':r['prompt'],'ground_truth':r['ground_truth'],
                      'prediction':text,'correct':value is not None and value==gold(r['ground_truth']),
                      'parsed':str(value) if value is not None else None,'method':method,
                      'tokens':len(ids),'hit_cap':len(ids)==a.max_tokens and ids[-1] not in stops}
                generated.append(item)
                with (out/f'{mode}.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            print(mode,len(generated),'/',len(rows),flush=True)
        scores=[int(r['correct']) for r in generated]
        result={'accuracy':sum(scores)/len(scores),'cap_rate':sum(r['hit_cap'] for r in generated)/len(scores),
                'mean_tokens':sum(r['tokens'] for r in generated)/len(scores),'unparsed_ids':[r['id'] for r in generated if r['parsed'] is None],
                'generation':dict(config,max_new_tokens=a.max_tokens,per_batch_seed='42+start'),'scores':scores}
        if a.baseline:
            baseline=[json.loads(x) for x in (Path(a.baseline)/f'{mode}.jsonl').read_text().splitlines()]
            assert [(r['id'],r['prompt'],r['ground_truth']) for r in baseline]==[(r['id'],r['prompt'],r['ground_truth']) for r in generated]
            result['vs_baseline']=paired([int(r['correct']) for r in baseline],scores)
            result['text_changed']=sum(r['prediction']!=b['prediction'] for r,b in zip(generated,baseline))
        summary['modes'][mode]=result
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('RESULT',mode,json.dumps(result),flush=True)
    summary.update(complete=True,end_time=time.time())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
