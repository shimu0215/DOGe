"""Pinned official MAWPS+ASDiv-A training, SVAMP never enters training buffers."""
import argparse,csv,hashlib,json,random,re,sys,io
from decimal import Decimal
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,'/scratch/wzhao20/AKDA2/gsm_vocab_aligned_minillm_20260909')
from data_utils.indexed_dataset import make_builder,best_fitting_dtype
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--student',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
g=ROOT/'results/generalization_20260913';data=g/'mawps_asdiv_official';m=json.loads((data/'manifest.json').read_text());raw=(data/'train.csv').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['sha256']
svamp_path=g/'svamp_headroom_22481_data/SVAMP.json';svamp=json.loads(svamp_path.read_text());assert len(svamp)==1000
num=lambda s:format(Decimal(s).normalize(),'f')
def canon(s):
 s=re.sub(r'\d+(?:\.\d+)?',lambda x:num(x.group()),s.lower());return re.sub(r'[^a-z0-9]','',s)
heldout={canon(r['Body']+' '+r['Question']) for r in svamp};seen=set();rows=[];overlap=[];duplicates=[]
for i,r in enumerate(csv.DictReader(io.StringIO(raw.decode()))):
 numbers=r['Numbers'].split();q=re.sub(r'\bnumber(\d+)\b',lambda x:num(numbers[int(x.group(1))]),r['Question']);assert not re.search(r'\bnumber\d+\b',q)
 key=canon(q)
 if key in heldout:overlap.append(i);continue
 if key in seen:duplicates.append(i);continue
 seen.add(key);rows.append(dict(id=i,question=q))
random.Random(1111).shuffle(rows);tok=AutoTokenizer.from_pretrained(a.student);eligible=[];excluded=[]
for r in rows:
 prompt=tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':r['question']}],tokenize=False,add_generation_prompt=True);ids=tok.encode(prompt,add_special_tokens=False)
 if len(ids)>256:excluded.append(r['id']);continue
 eligible.append(dict(id=r['id'],instruction=r['question'],input='',output='',prompt=prompt,token_ids=ids))
assert len(eligible)>=1100
summary=dict(complete=False,source=m,source_rows=len(list(csv.DictReader(io.StringIO(raw.decode())))),svamp_sha256=hashlib.sha256(svamp_path.read_bytes()).hexdigest(),excluded_exact_normalized_svamp_overlap=overlap,excluded_duplicates=duplicates,excluded_long=excluded,answer_tokens_in_training=False,max_prompt_length=256,scope='Official related MAWPS+ASDiv-A train; SVAMP-derived-family similarity may remain despite normalized exact removal; no SVAMP examples in train or internal valid buffers',splits={})
for split,selected in [('train',eligible[:1000]),('valid',eligible[1000:1100])]:
 builder=make_builder(str(out/(split+'_0.bin')),impl='mmap',dtype=best_fitting_dtype(len(tok)));records=[]
 for r in selected:
  ids=r['token_ids'];builder.add_item(torch.IntTensor(ids));records.append({k:v for k,v in r.items() if k!='token_ids'})
 builder.finalize(str(out/(split+'_0.idx')));(out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records));summary['splits'][split]=dict(count=len(records),ids=[r['id'] for r in records])
summary['complete']=True;(out/'manifest.json').write_text(json.dumps(summary,indent=2));print('Prepared1000train100internalvalid',len(overlap),'SVAMP exact matches excluded')
