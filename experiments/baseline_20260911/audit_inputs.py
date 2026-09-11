"""Read-only provenance and rescoring before the dedicated baseline GPU is free."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
REPO=Path('/scratch/wzhao20/DOGe-official')
OUT=ROOT/'results/baseline_20260911';OUT.mkdir(exist_ok=True,parents=True)
run='qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908'
sft=REPO/'outputs'/run/'sft_gsm_test200/gsm8k-results.json'
raw=REPO/'outputs/qwen2_5_0p5b_instruct_sft_7b_cot_gsm1000_correctonly_20260908/raw_gsm_test200/gsm8k-results.json'
paths={'raw':raw,'sft':sft,'clean_opd':ROOT/'results/repaired_clean_gsm200/gsm8k-results.json'}
data={k:json.loads(p.read_text()) for k,p in paths.items()}
key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
assert key(data['raw'])==key(data['sft'])==key(data['clean_opd'])
assert data['raw']['generation']==data['sft']['generation']==data['clean_opd']['generation']
scores={k:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']] for k,d in data.items()}
accuracy={k:sum(v)/len(v) for k,v in scores.items()}
prompts=Path('/scratch/wzhao20/AKDA2/opd_antidistill_minillm/experiments/qwen2_5_0p5b_7b_sft_then_14b_opd_gsm_chatprompt_20260908/prompts/train.jsonl')
rows=[json.loads(x) for x in prompts.read_text().splitlines()]
source=REPO/'data/gsm8k_train1000_qwen25_7b_qwen25_14b_greedy_20260908/qwen2.5-14b-instruct.jsonl'
source_rows=[json.loads(x) for x in source.read_text().splitlines()]
questions={r['instruction'] for r in rows}
# Check the CoT source identifiers really belong to the same first1000 pool.
source_keys=sorted(source_rows[0])
source_ids=[]
for r in source_rows:
    ident=r.get('source_id',r.get('id',r.get('dataset_index')))
    assert ident is not None,source_keys
    source_ids.append(ident)
assert all(isinstance(i,int) and 0<=i<1000 for i in source_ids),source_ids[:10]
assert {r['source_id'] for r in rows}==set(range(1000))
files=[prompts,source,REPO/'data'/run/'train.jsonl',REPO/'scripts/train_gsm_cot_sft.py',*paths.values()]
result=dict(paths={k:str(p) for k,p in paths.items()},models={k:d['model_name'] for k,d in data.items()},
    accuracy=accuracy,n=len(scores['raw']),sft_gain_pp=100*(accuracy['sft']-accuracy['raw']),
    target_pre_sft=accuracy['raw'],target_half_sft_gain=accuracy['raw']+.5*(accuracy['sft']-accuracy['raw']),
    scores=scores,sft_vs_raw=paired(scores['raw'],scores['sft']),
    generation=data['raw']['generation'],training_questions=sorted(questions),source_keys=source_keys,
    file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    note='Pre-SFT means before this project CoT SFT; source checkpoint is already vendor instruction-tuned. Targets are endpoint research goals, not retained-checkpoint adversarial guarantees.')
(OUT/'input_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['scores','training_questions','file_sha256','generation']},indent=2))
