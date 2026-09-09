"""Independently replay clean teacher batches to check the zero-hit control method."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


@torch.inference_mode()
def main():
    root=Path('results');teacher='/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct'
    tok=AutoTokenizer.from_pretrained(teacher);tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(teacher,torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
    eos=model.generation_config.eos_token_id
    eos=sorted(set((eos if isinstance(eos,list) else [eos])+[tok.pad_token_id,tok.eos_token_id]))
    result={}
    for name,sample in [('teacher_likelihood_gsm200',False),('teacher_likelihood_sampling64',True)]:
        rows=[json.loads(x) for x in (root/name/'rows.jsonl').read_text().splitlines()][:8]
        enc=tok([r['prompt'] for r in rows],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
        kwargs={'do_sample':sample}
        if sample:kwargs.update(temperature=.7,top_p=.8,top_k=20)
        torch.manual_seed(42)
        generated=model.generate(**enc,**kwargs,max_new_tokens=512,eos_token_id=eos,
            pad_token_id=tok.pad_token_id,use_cache=True)
        pred=tok.batch_decode(generated[:,enc.input_ids.size(1):],skip_special_tokens=True)
        matches=[p==r['prediction'] for p,r in zip(pred,rows)]
        result[name]={'independent_clean_replays':len(matches),'identical_text':sum(matches),
            'mismatch_ids':[r['id'] for r,ok in zip(rows,matches) if not ok]}
        print(name,result[name],flush=True)
        assert all(matches),'Zero-hit generation differed from independent clean replay'
    (root/'teacher_identity_replay.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
