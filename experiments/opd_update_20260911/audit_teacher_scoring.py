"""Read-only held-out prefix audit of compliant exported teachers.

Loads teacher checkpoints only. Student data are fixed offline token sequences.
This is a distribution diagnostic, not an estimate of student learning outcomes.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/rl_process_20260910'))
sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
from objectives import process_end
from teacher_only_poison import TeacherOnlyPermutation
from corrected_numeric_audit import prediction, gold

p = argparse.ArgumentParser()
p.add_argument('--teacher', required=True)
p.add_argument('--candidates', nargs='+', required=True)
p.add_argument('--context', required=True)
p.add_argument('--output', required=True)
p.add_argument('--examples', type=int, default=24)
a = p.parse_args()
assert os.environ.get('SLURM_JOB_ID')
assert torch.cuda.device_count() == 1
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
started = time.time()
tok = AutoTokenizer.from_pretrained(a.teacher)
context = Path(a.context)
rows = [json.loads(x) for x in (context / 'rollouts.jsonl').read_text().splitlines()]
ids = sorted({r['example_id'] for r in rows if r['source'] == 'student'})
assert len(ids) == 384 and 0 < a.examples <= 64
heldout = ids[-64:][:a.examples]
raw_ids = set(json.loads((context/'manifest.json').read_text()).get('raw_replacement_ids', []))
chosen_rows = []
for row in rows:
    if row['example_id'] not in heldout or row['source'] not in ['student', 'teacher', 'teacher_greedy']:
        continue
    stop = process_end(tok, row['response_ids'])
    available = [j for j in range(32, stop) if row['response_ids'][j] not in tok.all_special_ids]
    if not available:
        continue
    seed = 17000 + ids.index(row['example_id'])
    offsets = sorted(random.Random(seed).sample(available, min(32, len(available))))
    chosen_rows.append((row, offsets))
assert chosen_rows
manifest = dict(vars(a), complete=False, start=started, heldout_ids=heldout,
    student_model_loaded=False, student_parameter_signal=False, student_outcome_reward=False,
    context_sha256=hashlib.sha256((context / 'rollouts.jsonl').read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    note='Post-export heldout teacher distributions; not student performance or a training reward.', results={})

def save():
    tmp = out / 'manifest.tmp'
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out / 'manifest.json')

@torch.no_grad()
def lp_at(model, row, offsets):
    ids_tensor = torch.tensor([row['prompt_ids'] + row['response_ids'][:max(offsets)+1]], device='cuda')
    positions = [len(row['prompt_ids'])-1+j for j in offsets]
    hidden = model.model(input_ids=ids_tensor, use_cache=False).last_hidden_state[0, positions]
    return model.lm_head(hidden).float().log_softmax(-1)

def load_model(path):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()

save()
reference = load_model(a.teacher)
permutation = TeacherOnlyPermutation(tok, 'permute_topk', k=2)
for candidate_path in a.candidates:
    provenance = json.loads((Path(candidate_path).parent / 'manifest.json').read_text())
    assert provenance['complete'] and provenance['plain_export_verified']
    assert provenance['student_model_loaded'] is False and provenance['student_parameter_signal'] is False
    assert provenance['student_outcome_reward'] is False and provenance['teacher'] == a.teacher
    label = Path(candidate_path).parent.name
    assert label not in manifest['results']
    candidate = load_model(candidate_path)
    totals = {}
    with (out / (label + '_rows.jsonl')).open('x') as stream, torch.no_grad():
        for row, offsets in chosen_rows:
            original = lp_at(reference, row, offsets)
            current = lp_at(candidate, row, offsets)
            target = permutation(original).log_softmax(-1)
            eligible = original.clone()
            eligible[:, tok.all_special_ids] = -torch.inf
            eligible[:, len(tok):] = -torch.inf
            top = eligible.topk(2, dim=-1)
            ratio = (top.values[:, 0]-top.values[:, 1]).exp()
            bounded = (top.values[:, 1].exp() >= .02) & (top.values[:, 0].exp() >= .35) & (ratio >= 1.5) & (ratio <= 20.)
            divergence = (target.exp()*(target-original)).sum(-1)
            largest = torch.zeros(len(offsets), dtype=torch.bool, device='cuda')
            largest[divergence.topk(min(8, len(offsets))).indices] = True
            wanted = target.argmax(-1, keepdim=True)
            observed = torch.tensor([row['response_ids'][j] for j in offsets], device='cuda')[:, None]
            metrics = dict(original_entropy=-(original.exp()*original).sum(-1),
                current_entropy=-(current.exp()*current).sum(-1),
                original_observed_logp=original.gather(-1, observed).squeeze(-1),
                current_observed_logp=current.gather(-1, observed).squeeze(-1),
                reference_kl=(original.exp()*(original-current)).sum(-1),
                target_kl_before=divergence, target_kl_after=(target.exp()*(target-current)).sum(-1),
                original_target_probability=original.gather(-1, wanted).exp().squeeze(-1),
                current_target_probability=current.gather(-1, wanted).exp().squeeze(-1),
                argmax_flip=(current.argmax(-1) != original.argmax(-1)).float(),
                target_argmax=(current.argmax(-1) == wanted.squeeze(-1)).float(),
                observed_token_logp_change=(current.gather(-1, observed)-original.gather(-1, observed)).squeeze(-1))
            answer, method = prediction(row['text'].replace(r'\,', ' '))
            outcome = 'incomplete' if row.get('hit_cap') or method != 'boxed' else ('correct' if answer == gold(row['gold']) else 'incorrect')
            origin = ('raw' if row['example_id'] in raw_ids else 'fullsft') if row['source'] == 'student' else row['source']
            for group, mask in [('origin_'+origin, torch.ones_like(bounded)),
                                ('origin_'+origin+'_'+outcome, torch.ones_like(bounded)), ('all', torch.ones_like(bounded)), ('bounded', bounded), ('largest_target_kl', largest)]:
                count = int(mask.sum())
                if not count:
                    continue
                values = {key: float(value[mask].sum()) for key, value in metrics.items()}
                key = row['source'] + '/' + group
                acc = totals.setdefault(key, dict(positions=0, rows=0, sums={k: 0. for k in metrics}))
                acc['positions'] += count
                acc['rows'] += 1
                for k, v in values.items():
                    acc['sums'][k] += v
                stream.write(json.dumps(dict(example_id=row['example_id'], source=row['source'], origin=origin, outcome=outcome, group=group,
                    positions=count, means={k: v/count for k, v in values.items()}))+'\n')
    manifest['results'][label] = {key: dict(positions=v['positions'], rows=v['rows'],
        means={k: x/v['positions'] for k, x in v['sums'].items()}) for key, v in totals.items()}
    save()
    del candidate
    gc.collect()
    torch.cuda.empty_cache()
manifest.update(complete=True, end=time.time())
save()
print(json.dumps(manifest), flush=True)
