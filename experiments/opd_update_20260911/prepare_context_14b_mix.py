"""Broaden fixed negative texts; no generator weights/logits are loaded."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
base=ROOT/'results/opd_update_20260911/static_mixctx_top2_9870980_context';m=json.loads((base/'manifest.json').read_text());assert m['complete']
ps=Path(m['prompts']);assert hashlib.sha256(ps.read_bytes()).hexdigest()==m['prompt_sha256'];prompts=[json.loads(s) for s in ps.read_text().splitlines()]
rows=[json.loads(s) for s in (base/'rollouts.jsonl').read_text().splitlines()];neg={r['example_id']:r for r in rows if r['source']=='student'};assert len(neg)==384
tok=AutoTokenizer.from_pretrained(m['teacher'])
source=Path('/scratch/wzhao20/DOGe-official/data/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/train.jsonl')
texts={};rejected=[]
for row in [json.loads(s) for s in source.read_text().splitlines()]:
 n=next(i for i,x in enumerate(row['labels']) if x!=-100);assert all(x==y for x,y in zip(row['labels'][n:],row['input_ids'][n:]))
 prompt=tok.decode(row['input_ids'][:n],skip_special_tokens=False);question=prompt.split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
 texts[int(row['id'])]=(question,tok.decode(row['input_ids'][n:],skip_special_tokens=True))
available={}
for i,row in neg.items():
 example=prompts[row['dataset_index']];assert tok.encode(example['prompt'],add_special_tokens=False)==row['prompt_ids']
 source_id=example['source_id'];assert source_id==row['source_id'] and source_id<1000
 if source_id not in texts:continue
 question,text=texts[source_id];assert question==example['instruction']
 pred=prediction(text.replace(chr(92)+',',' '))[0];expected=gold(example['output'])
 if pred!=expected:rejected.append(dict(source_id=source_id,predicted=str(pred),expected=str(expected)));continue
 ids=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id];available[i]=ids
ids=sorted(neg);tr,va=ids[:-64],ids[-64:];pooltr=[i for i in tr if i in available];poolva=[i for i in va if i in available];assert len(pooltr)>=160 and len(poolva)>=32
rng=random.Random(20260913);selected=set(rng.sample(pooltr,160)+rng.sample(poolva,32));changed=[]
for row in rows:
 if row['source']=='student' and row['example_id'] in selected:
  response=available[row['example_id']];row=dict(row,response_ids=response[:384],text=tok.decode(response[:384],skip_special_tokens=True),hit_cap=len(response)>384,negative_generator='fixed 14B CoT text from existing SFT corpus; no weights loaded');changed.append(row['example_id'])
 with (out/'rollouts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
assert len(changed)==192
m['original_raw_replacement_ids']=m['raw_replacement_ids'];m['raw_replacement_ids']=[i for i in m['raw_replacement_ids'] if i not in selected]
m['original_sampling']=m['sampling'];m['sampling']={'remaining_0p5b':m['sampling'],'added14b':'Historical fixed CoT corpus; no new sampling performed'}
m.update(output=str(out),source_context=str(base),source_context_sha256=hashlib.sha256((base/'rollouts.jsonl').read_bytes()).hexdigest(),source_14b_text=str(source),source_14b_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),student=m['student']+['fixed offline14B CoT texts (weights not loaded)'],complete=True,offline_sources_only=True,selection='Fixed 160 train and32heldout replacements among exact-matched, numerically-correct14B texts; original teacher positive rows unchanged, max384prefix budget unchanged',replacement_ids=sorted(changed),replacement_train_count=160,replacement_validation_count=32,rejected_source_answers=rejected,eligible_train=len(pooltr),eligible_validation=len(poolva),generator_models_loaded=False,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),end=time.time())
(out/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m))
