"""Matched frozen-teacher diagnostic on 32 held-out offline questions; no training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--job', required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = 'teacher_repair_matched_' + a.job
record = OUT / (tag + '_worker.json')
assert not record.exists(), record
state = dict(start=time.time(), job=a.job, complete=False, completed=[],
             method_scope='No student model or parameter-derived signal in teacher training; fixed negative CoTs only')

def read(path):
    return json.loads(path.read_text())

def save():
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(record)

def run(phase, cmd, env=None):
    state.update(phase=phase, command=cmd)
    save()
    with (OUT / (tag + '_' + phase + '.log')).open('x') as log:
        child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        save()
        code = child.wait()
    if code:
        raise RuntimeError(phase + ' exit=' + str(code))
    state['completed'].append(phase)
    save()

try:
    save()
    assert os.environ['SLURM_JOB_ID'] == a.job
    info = subprocess.check_output(['scontrol', 'show', 'job', a.job, '-o'], text=True)
    fields = dict(x.split('=', 1) for x in info.split() if '=' in x)
    assert fields['JobState'] == 'RUNNING'
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 600
    import torch
    assert torch.cuda.device_count() == 1
    devices = [line.strip().split(', ') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).strip().splitlines()]
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    assert ',' not in visible
    if len(devices) == 1:
        device_uuid = devices[0][1]
    else:
        matched = [uuid for index, uuid in devices if index == visible or uuid == visible]
        assert len(matched) == 1, (visible, devices)
        device_uuid = matched[0]
    assert not subprocess.check_output(['nvidia-smi', '-i', device_uuid,
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    state['allocation'] = info
    state['device'] = dict(visible=os.environ['CUDA_VISIBLE_DEVICES'], step=os.environ['SLURM_STEP_ID'],
                          uuid=device_uuid)
    save()
    import gc,random,hashlib
    from transformers import AutoModelForCausalLM,AutoTokenizer
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    sys.path.insert(0,str(ROOT/'experiments/rl_process_20260910'))
    from objectives import process_end
    from teacher_only_poison import TeacherOnlyPermutation
    from teacher_only_tail_permutation_v2 import TailPermutation
    context=OUT/'static_mixctx_top2_9870980_context'
    reference_manifest=read(OUT/'static_tail_last2_anchor24_9983837/manifest.json')
    eligible=reference_manifest['training_prompt_ids'];heldout=reference_manifest['validation_prompt_ids']
    order=random.Random(1111).sample(eligible,len(eligible))
    chosen={'trained':set(),'heldout':set(heldout[:32])}
    assert not (chosen['trained'] & chosen['heldout'])
    tok=AutoTokenizer.from_pretrained(os.environ['TEACHER'])
    rows=[]
    for line in (context/'rollouts.jsonl').read_text().splitlines():
        row=json.loads(line);label=next((k for k,v in chosen.items() if row['example_id'] in v),None)
        if label is None:continue
        row['split']=label;row['process_end']=process_end(tok,row['response_ids'])
        if row['process_end']>32:rows.append(row)
    targets={'top2':TeacherOnlyPermutation(tok,'permute_topk',k=2),'tail':TailPermutation(tok,count=16)}
    models=[('hardfreq','static_tail_frequent_hardrl_9983835','tail'),('anchor36','static_tail_freq_anti6_anchor36_9916493','tail'),('answer4','static_tail_freq_anti6_answer4_9916493','tail')]
    state.update(training_performed=False,method_scope='Read-only frozen teacher target-fit diagnostic on offline text; not a deployable defense or source classifier',
        selection='All up-to32 uniformly sampled process positions per prefix, no targetKL ranking. Disagreement is originalteacher argmax versus offline token ID, not mathematical error.',
        selected_question_ids={k:sorted(v) for k,v in chosen.items()},rows=len(rows),
        context_sha256=hashlib.sha256((context/'rollouts.jsonl').read_bytes()).hexdigest(),models=models)
    save()
    ref=AutoModelForCausalLM.from_pretrained(os.environ['TEACHER'],torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval()
    def lp(model,row,offsets):
        ids=torch.tensor([row['prompt_ids']+row['response_ids'][:max(offsets)+1]],device='cuda')
        hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0]
        indices=[len(row['prompt_ids'])-1+j for j in offsets]
        return model.lm_head(hidden[indices]).float().log_softmax(-1)
    metrics=[]
    detail_path=OUT/(tag+'_rows.jsonl');assert not detail_path.exists()
    for name,source,target_kind in models:
        if datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()<180:
            state.setdefault('models_skipped_budget',[]).append(name);save();continue
        manifest=read(OUT/source/'manifest.json');assert manifest['complete'] and manifest['plain_export_verified']
        assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
        state.update(phase='fit_'+name);save()
        candidate=AutoModelForCausalLM.from_pretrained(OUT/source/'model',torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval()
        for i,row in enumerate(rows):
            assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()>=30,'Stop before allocation expiry'
            available=[j for j in range(32,row['process_end']) if row['response_ids'][j] not in tok.all_special_ids]
            if not available:continue
            offsets=sorted(random.Random(710000+row['example_id']).sample(available,min(32,len(available))))
            with torch.no_grad():
                original=lp(ref,row,offsets);target=targets[target_kind](original).log_softmax(-1)
                divergence=(target.exp()*(target-original)).sum(-1)
                indices=torch.arange(len(offsets),device='cuda')
                selected=[offsets[j] for j in indices.tolist()]
                original=original[indices];target=target[indices];current=lp(candidate,row,selected)
                base_kl=float((target.exp()*(target-original)).sum(-1).mean())
                fit_kl=float((target.exp()*(target-current)).sum(-1).mean())
                native_kl=float((original.exp()*(original-current)).sum(-1).mean())
                observed=torch.tensor([row['response_ids'][j] for j in selected],device='cuda')[:,None]
                disagreement=original.argmax(-1)!=observed.squeeze(-1)
                observed_change=(current.gather(-1,observed)-original.gather(-1,observed)).squeeze(-1)
                entry=dict(n_positions=len(selected),n_disagreement=int(disagreement.sum()),
                    disagreement_observed_delta_sum=float(observed_change[disagreement].sum()),
                    agreement_observed_delta_sum=float(observed_change[~disagreement].sum()),
                    original_top_ids=original.argmax(-1).tolist(),observed_ids=observed.squeeze(-1).tolist(),
                    model=name,source=row['source'],split=row['split'],example_id=row['example_id'],positions=selected,
                    target_kl_original=base_kl,target_kl_candidate=fit_kl,reference_kl=native_kl,
                    relative_target_loss_reduction=(base_kl-fit_kl)/base_kl if base_kl>1e-6 else None,
                    argmax_flip=float((original.argmax(-1)!=current.argmax(-1)).float().mean()),
                    entropy_change=float((-(current.exp()*current).sum(-1)+(original.exp()*original).sum(-1)).mean()),
                    observed_logp_change=float((current.gather(-1,observed)-original.gather(-1,observed)).mean()),
                    original_top_tokens=[tok.decode([j]) for j in original.argmax(-1).tolist()],
                    target_top_tokens=[tok.decode([j]) for j in target.argmax(-1).tolist()])
            metrics.append(entry)
            with detail_path.open('a') as f:f.write(json.dumps(entry)+'\n')
            if i%32==0:state['rows_scored']=len(metrics);save()
        del candidate;gc.collect();torch.cuda.empty_cache()
        state['completed'].append(name);save()
    summaries={}
    for m in metrics:
        summaries.setdefault((m['model'],m['source'],m['split']),[]).append(m)
    state['summary']=[dict(model=k[0],source=k[1],split=k[2],n=len(v),
        **{field:sum(x[field] for x in v)/len(v) for field in ['target_kl_original','target_kl_candidate','reference_kl','argmax_flip','entropy_change','observed_logp_change']}) for k,v in summaries.items()]
    state['conditional_observed_changes']=[dict(model=k[0],source=k[1],split=k[2],positions=sum(x['n_positions'] for x in v),disagreements=sum(x['n_disagreement'] for x in v),disagreement_delta_sum=sum(x['disagreement_observed_delta_sum'] for x in v),agreement_delta_sum=sum(x['agreement_observed_delta_sum'] for x in v)) for k,v in summaries.items()]
    state.update(complete=True,phase='complete',rows_scored=len(metrics),end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
