"""Test normal teacher greedy generation with the same gate used by OPD.

Untouched greedy trajectories are their own exact clean control by causal induction.
Only trajectories on which gate actually changes logits need a separate clean decode.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,LogitsProcessorList
from doge.evaluation import evaluate_predictions
from likelihood_gate import FixedReferenceGate,GenerationGate


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--examples',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=200);p.add_argument('--batch',type=int,default=8)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads(Path(args.examples).read_text())['content'][:args.limit]
    tokenizer=AutoTokenizer.from_pretrained(args.teacher);tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(args.teacher,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
    configured=model.generation_config.eos_token_id
    eos=sorted(set(([configured] if isinstance(configured,int) else configured)+[tokenizer.pad_token_id,tokenizer.eos_token_id]))
    gate=FixedReferenceGate(args.reference,tokenizer.pad_token_id,eos,alpha=.01,margin=2.)
    original=model._get_logits_processor
    def gate_first(*a,**kw):
        processors=original(*a,**kw)
        return LogitsProcessorList([x for x in processors if isinstance(x,GenerationGate)]+[x for x in processors if not isinstance(x,GenerationGate)])
    model._get_logits_processor=gate_first
    results=[];n_seen=0;n_hit=0
    torch.manual_seed(42)
    for start in range(0,len(rows),args.batch):
        batch=rows[start:start+args.batch]
        enc=tokenizer([r['prompt'] for r in batch],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
        proc=GenerationGate(gate)
        output=model.generate(**enc,do_sample=False,max_new_tokens=512,eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id,use_cache=True,logits_processor=LogitsProcessorList([proc]))
        response=output[:,enc.input_ids.size(1):]
        texts=tokenizer.batch_decode(response,skip_special_tokens=True)
        hit=proc.ever.cpu().tolist();n_seen+=proc.n_seen;n_hit+=proc.n_hit
        controls=list(texts)
        changed=[i for i,x in enumerate(hit) if x]
        if changed:
            cc=tokenizer([batch[i]['prompt'] for i in changed],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
            clean=model.generate(**cc,do_sample=False,max_new_tokens=512,eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,use_cache=True)
            for i,t in zip(changed,tokenizer.batch_decode(clean[:,cc.input_ids.size(1):],skip_special_tokens=True)):controls[i]=t
        for j,r in enumerate(batch):
            result={'id':r['id'],'prompt':r['prompt'],'ground_truth':r['ground_truth'],
                'prediction':texts[j],'clean_prediction':controls[j],'gate_ever':hit[j],
                'response_ids':response[j].cpu().tolist()}
            results.append(result)
            with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print(f'teacher greedy {len(results)}/{len(rows)}; triggered sequences={sum(r["gate_ever"] for r in results)}',flush=True)
        del proc
    references=[r['ground_truth'] for r in results]
    summary={'teacher':args.teacher,'reference':args.reference,'n':len(results),
        'generation':{'do_sample':False,'max_new_tokens':512,'seed':42,'inherits_teacher_repetition_penalty':model.generation_config.repetition_penalty},
        'gate':{'alpha':.01,'threshold':gate.threshold,'latch':True,'margin':gate.margin,'sharp':gate.sharp,'EOS_protection':True},
        'gated':evaluate_predictions([r['prediction'] for r in results],references),
        'clean':evaluate_predictions([r['clean_prediction'] for r in results],references),
        'token_trigger_rate':n_hit/max(n_seen,1),'triggered_sequences':sum(r['gate_ever'] for r in results),
        'text_changed_sequences':sum(r['prediction']!=r['clean_prediction'] for r in results),
        'control_method':'same greedy trajectory when zero changes; separate clean generation for every triggered sequence'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
