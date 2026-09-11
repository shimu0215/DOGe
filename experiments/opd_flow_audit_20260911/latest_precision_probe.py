"""Measure teacher precision and defense signals on the same live OPD prefixes."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/opd_flow_audit_20260911'
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from repair_runtime import full_attention_mask


def sim(a,b):
    a=a.flatten().double();b=b.flatten().double()
    return dict(norm_a=float(a.norm()),norm_b=float(b.norm()),
        norm_ratio=float(a.norm()/b.norm()) if b.norm()>0 else None,
        cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)) if a.norm()>0 and b.norm()>0 else None)


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    destination=OUT/'latest_precision_probe.json';assert not destination.exists()
    rows=json.loads((OUT/'live_rollouts.json').read_text())
    tok=AutoTokenizer.from_pretrained(os.environ['PROXY'])
    tok.pad_token_id=tok.eos_token_id
    result=dict(start=time.time(),context_sha256=hashlib.sha256((OUT/'live_rollouts.json').read_bytes()).hexdigest(),
        scope='Same four fresh SFT-student rollouts, up to16 evenly spaced valid response positions each. No performance or full-OPD efficacy claim.',models={},skipped=[])
    cache=[]
    student=AutoModelForCausalLM.from_pretrained(os.environ['PROXY'],torch_dtype=torch.bfloat16,device_map='cuda').eval()
    for row in rows:
        positions=[i for i,m in enumerate(row['mask']) if m]
        select=sorted(set(positions[int(j*(len(positions)-1)/15)] for j in range(16)))
        q=torch.tensor([row['prompt_ids']],device='cuda')
        r=torch.tensor([row['response_ids'][:max(select)+1]],device='cuda')
        ids=torch.cat((q,r),-1);mask=full_attention_mask(q,r,tok.pad_token_id)
        idx=[q.size(1)-1+j for j in select]
        h=student.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state[0,idx]
        lp=student.lm_head(h).float().log_softmax(-1).cpu()
        cache.append(dict(ids=ids.cpu(),mask=mask.cpu(),idx=idx,lp=lp,
                          selected=torch.tensor([row['response_ids'][j] for j in select])))
    del student,h,lp;gc.collect();torch.cuda.empty_cache()
    paths=[('original',Path(os.environ['TEACHER']))]+[(name,ROOT/'results/opd_update_20260911'/name/'model')
        for name in ['direct_rank_strong','direct_fkl','direct_fkl_dense']]
    originals={};baseline_weights={}
    for name,path in paths:
        if not path.exists():result['skipped'].append(name);continue
        by_dtype={};edit_deltas={};weight_deltas={}
        for dtype in [torch.float16,torch.bfloat16]:
            label=str(dtype)
            model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=dtype,device_map='cuda').eval()
            values=[];outside=[]
            for row in cache:
                h=model.model(input_ids=row['ids'].cuda(),attention_mask=row['mask'].cuda(),use_cache=False).last_hidden_state[0,row['idx']]
                z=model.lm_head(h).float();v=row['lp'].size(-1)
                lp=z[:,:v].log_softmax(-1).cpu()
                outside.append(float((1-z.log_softmax(-1)[:,:v].exp().sum(-1)).clamp_min(0).mean()))
                values.append(lp)
            teacher_lp=torch.cat(values);student_lp=torch.cat([r['lp'] for r in cache])
            ids=torch.cat([r['selected'] for r in cache])[:,None]
            weights=torch.cat([p.flatten()[::max(1,p.numel()//4096)].float().cpu()
                for n,p in model.named_parameters()
                if any(n.startswith('model.layers.'+str(i)+'.') for i in range(model.config.num_hidden_layers-4,model.config.num_hidden_layers))
                and n.endswith(('q_proj.weight','v_proj.weight','down_proj.weight'))])
            if name=='original':originals[label]=teacher_lp;baseline_weights[label]=weights
            delta=teacher_lp-originals[label];observed=delta.gather(-1,ids).squeeze(-1)
            weight_deltas[label]=weights-baseline_weights[label]
            edit_deltas[label]=observed
            by_dtype[label]=dict(actual_dtype=str(next(model.parameters()).dtype),
                conditional_reverse_kl=float((student_lp.exp()*(student_lp-teacher_lp)).sum(-1).mean()),
                probability_outside_student_head=sum(outside)/len(outside),
                edit_observed_logprob_rms=float(observed.square().mean().sqrt()),
                edit_student_weighted_logprob_rms=float((student_lp.exp()*delta.square()).sum(-1).mean().sqrt()),
                sampled_weight_delta_norm=float(weight_deltas[label].norm()),
                sampled_changed_weight_fraction=float(weight_deltas[label].ne(0).float().mean()))
            del model,h,z,lp,values;gc.collect();torch.cuda.empty_cache()
        result['models'][name]=dict(path=str(path),by_dtype=by_dtype,
            bf16_vs_fp16_edit_signal=sim(edit_deltas['torch.bfloat16'],edit_deltas['torch.float16']),
            bf16_vs_fp16_weight_delta=sim(weight_deltas['torch.bfloat16'],weight_deltas['torch.float16']))
        if name=='original':
            delta=originals['torch.bfloat16']-originals['torch.float16']
            obs=delta.gather(-1,ids).squeeze(-1)
            result['original_precision_difference']=dict(observed_logprob_rms=float(obs.square().mean().sqrt()),
                student_weighted_logprob_rms=float((student_lp.exp()*delta.square()).sum(-1).mean().sqrt()))
        (OUT/'latest_precision_progress.json').write_text(json.dumps(result,indent=2))
        print('PRECISION',name,json.dumps(result['models'][name]),flush=True)
    result.update(complete=True,end=time.time())
    destination.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
