"""Create fixed synthetic negative contexts using original-teacher CoTs only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/gate_audit_20260909'))
sys.path.insert(0, str(ROOT/'experiments/rl_process_20260910'))
from corrected_numeric_audit import prediction, gold
from objectives import process_end

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    src, out=Path(a.source), Path(a.output)
    cm=json.loads((src/'manifest.json').read_text())
    assert cm['complete']
    rows=[json.loads(line) for line in (src/'rollouts.jsonl').read_text().splitlines()]
    own=[r for r in rows if r['source'] in ['teacher','teacher_greedy']]
    ids=sorted({r['example_id'] for r in own})
    assert len(ids)==384 and len(own)==768
    prompts_path=Path(cm['prompts'])
    assert hashlib.sha256(prompts_path.read_bytes()).hexdigest()==cm['prompt_sha256']
    prompts=[json.loads(line) for line in prompts_path.read_text().splitlines()]
    tok=AutoTokenizer.from_pretrained(cm['teacher'])
    digit_ids={str(i):tok.encode(str(i),add_special_tokens=False) for i in range(10)}
    assert all(len(v)==1 for v in digit_ids.values())
    single_ids={v[0] for v in digit_ids.values()}
    preferred={}; fallback={}
    for row in own:
        i=row['example_id']
        if i not in fallback or row['source']=='teacher_greedy': fallback[i]=row
        value,method=prediction(tok.decode(row['response_ids'],skip_special_tokens=True).replace(r'\,',' '))
        correct=not row.get('hit_cap') and method=='boxed' and value==gold(prompts[row['dataset_index']]['output'])
        if correct and (i not in preferred or row['source']=='teacher_greedy'): preferred[i]=row
    negatives=[]; changed=[]; eligible=[]
    for i in ids:
        origin=preferred.get(i,fallback[i])
        row=copy.deepcopy(origin); tokens=row['response_ids']
        end=process_end(tok,tokens)
        candidates=[j for j in range(32,min(end//2,end-24)) if tokens[j] in single_ids]
        row.update(source='synthetic_teacher', origin_source=origin['source'],
                   corruption_end=None, corruption=None)
        if candidates:
            rng=random.Random(1981+i); j=rng.choice(candidates)
            old=tokens[j]; alternatives=sorted(single_ids-{old}); new=rng.choice(alternatives)
            tokens[j]=new
            row.update(corruption_end=j+1,corruption=dict(position=j,original_id=old,replacement_id=new),
                       text=tok.decode(tokens,skip_special_tokens=True))
            assert tokens[:j]==origin['response_ids'][:j] and tokens[j+1:]==origin['response_ids'][j+1:]
            changed.append(i)
            if i in preferred and end>j+9 and i in ids[:-64]: eligible.append(i)
        negatives.append(row)
    assert len(eligible)>64, len(eligible)
    out.mkdir(parents=True,exist_ok=False)
    path=out/'rollouts.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in negatives+own))
    m=dict(teacher=cm['teacher'],student=None,prompts=cm['prompts'],prompt_sha256=cm['prompt_sha256'],
           complete=True,n=384,source_context=str(src),source_context_sha256=hashlib.sha256((src/'rollouts.jsonl').read_bytes()).hexdigest(),
           code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           context_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),negative_generator='Original teacher offline CoT with one deterministic digit substitution',
           student_generated_rows_used=False,student_model_loaded=False,student_parameter_signal=False,
           mutated_ids=changed,eligible_training_ids=eligible,heldout_ids=ids[-64:],
           policy='Negative loss only after corruption_end+4, so the scored prefix includes the mutation. No source label added to model input.',
           limitation='A digit substitution is not guaranteed to make a trajectory incorrect. This is synthetic context augmentation, not a source-classification or defense guarantee.',
           end=time.time())
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
    print(json.dumps(dict(complete=True,rows=1152,mutated=len(changed),eligible_training=len(eligible),student_generated_rows_used=False)),flush=True)

if __name__=='__main__': main()
