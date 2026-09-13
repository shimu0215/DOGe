"""Frozen teacher distribution changes on matched, previously generated QA prefixes."""
import argparse,gc,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=c.Worker(a.job,'source_target_fit_'+a.job,minimum=180)
try:
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 sys.path.insert(0,str(ROOT/'experiments/opd_update_20260911'));from oracle_entropy4_target import TailPermutation
 sys.path.insert(0,str(ROOT/'experiments/rl_process_20260910'));from objectives import process_end
 sources={'self':'base_quarter_22479_original_teacher_val','coder':'coder_raw_high_28528_initial_val','main_short_sft':'../baseline_20260911/short_checkpoint-49_val'}
 docs={k:c.read(c.OUT/v/'gsm8k-results.json') for k,v in sources.items()}
 key=lambda d:[(r['id'],r['ground_truth']) for r in d['content'][:16]]
 assert all(key(d)==key(docs['self']) for d in docs.values())
 tok=AutoTokenizer.from_pretrained(c.ORIGINAL);target_fn=TailPermutation(tok);examples=[]
 for role,d in docs.items():
  for own,row in zip(docs['self']['content'][:16],d['content'][:16]):
   prompt=tok.encode(own['prompt'],add_special_tokens=False);response=tok.encode(row['prediction'],add_special_tokens=False)[:512]
   end=process_end(tok,response);available=[j for j in range(32,end) if response[j] not in tok.all_special_ids]
   if not available:continue
   offsets=sorted(random.Random(20000+row['id']).sample(available,min(32,len(available))))
   examples.append(dict(role=role,id=row['id'],prompt=prompt,response=response,offsets=offsets,correct=int(c.prediction(row['prediction'].replace(chr(92)+',',' '))[0]==c.gold(row['ground_truth']))))
 w.state.update(scope='Descriptive frozen-teacher changes on first16 matched QA ids; counterfactual entropy4 target applied identically for diagnostic only, including self and common teacher prompt. Existing source texts only; no source-classification accuracy, teacher training, or causal style claim.',sources={k:dict(path=str(c.OUT/v/'gsm8k-results.json'),sha256=hashlib.sha256((c.OUT/v/'gsm8k-results.json').read_bytes()).hexdigest(),generation=docs[k]['generation']) for k,v in sources.items()},examples=len(examples));w.save()
 ref=AutoModelForCausalLM.from_pretrained(c.ORIGINAL,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
 measurements=[]
 for label,source in [('entropy4','static_entropy4_dense_23371'),('pcgrad','static_entropy4_pcgrad_22478'),('mix14b','static_entropy4_mix14b_23371')]:
  path=ROOT/'results/opd_update_20260911'/source;m=c.read(path/'manifest.json');assert m['complete'] and m['plain_export_verified'] and not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
  model=AutoModelForCausalLM.from_pretrained(path/'model',torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval();w.state['phase']='score_'+label;w.save()
  for ex in examples:
   ids=torch.tensor([ex['prompt']+ex['response'][:max(ex['offsets'])+1]],device='cuda');positions=[len(ex['prompt'])-1+j for j in ex['offsets']]
   observed=torch.tensor([ex['response'][j] for j in ex['offsets']],device='cuda')[:,None]
   with torch.no_grad():
    def lp(which):return which.lm_head(which.model(input_ids=ids,use_cache=False).last_hidden_state[0,positions]).float().log_softmax(-1)
    native=lp(ref);changed=lp(model);target=target_fn(native).log_softmax(-1)
    target_native=float((target.exp()*(target-native)).sum(-1).mean());target_changed=float((target.exp()*(target-changed)).sum(-1).mean())
    row=dict(target_native_kl=target_native,target_changed_kl=target_changed,target_kl_fraction_closed=1-target_changed/max(target_native,1e-12),teacher=label,role=ex['role'],id=ex['id'],correct=ex['correct'],n_positions=len(positions),tokens=len(ex['response']),forward_kl=float((native.exp()*(native-changed)).sum(-1).mean()),observed_logp_change=float((changed.gather(-1,observed)-native.gather(-1,observed)).mean()),argmax_flip_fraction=float((native.argmax(-1)!=changed.argmax(-1)).float().mean()))
   measurements.append(row)
  del model;gc.collect();torch.cuda.empty_cache()
 summary={}
 for row in measurements:summary.setdefault(row['teacher'],{}).setdefault(row['role'],[]).append(row)
 summary={t:{s:dict(n=len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in ['forward_kl','observed_logp_change','argmax_flip_fraction','target_native_kl','target_changed_kl','target_kl_fraction_closed']},correct=sum(r['correct'] for r in rows)) for s,rows in roles.items()} for t,roles in summary.items()}
 c.write(c.OUT/(w.tag+'_measurements.json'),dict(summary=summary,rows=measurements));w.state['summary']=summary;w.state['limitations']='Different source generation settings, correctness, contents and lengths remain; not an isolated style/source-origin experiment. Distribution change is not OPD efficacy.';w.finish()
except Exception as e:w.fail(e);raise
