"""Native ChatML MATH train prompts, no answer tokens and no test overlap."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/scratch/wzhao20/AKDA2/gsm_vocab_aligned_minillm_20260909')
from data_utils.indexed_dataset import make_builder,best_fitting_dtype
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--student',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
trainpath=ROOT/'AgentDistill/data_processor/math_dataset/train/math_1000_20250414.json';testpath=ROOT/'AgentDistill/data_processor/math_dataset/test/math_500_20250414.json'
train=json.loads(trainpath.read_text())['examples'];test=json.loads(testpath.read_text())['examples'];assert len(train)==1000 and len(test)==500
assert len({r['question'] for r in train})==1000 and not {r['question'] for r in train}&{r['question'] for r in test}
tok=AutoTokenizer.from_pretrained(a.student);dtype=best_fitting_dtype(len(tok));summary={'complete':False,'source_train':str(trainpath),'train_sha256':hashlib.sha256(trainpath.read_bytes()).hexdigest(),'source_test':str(testpath),'test_sha256':hashlib.sha256(testpath.read_bytes()).hexdigest(),'no_train_test_overlap':True,'max_prompt_length':512,'answer_tokens_in_training':False,'splits':{}}
for split,rows in [('train',train),('valid',test[:100])]:
 builder=make_builder(str(out/(split+'_0.bin')),impl='mmap',dtype=dtype);records=[];excluded=[]
 for row in rows:
  prompt=tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},{'role':'user','content':row['question']}],tokenize=False,add_generation_prompt=True);ids=tok.encode(prompt,add_special_tokens=False)
  if len(ids)>512:excluded.append(row['id']);continue
  builder.add_item(torch.IntTensor(ids));records.append(dict(id=row['id'],instruction=row['question'],input='',output='',prompt=prompt,prompt_tokens=len(ids)))
 builder.finalize(str(out/(split+'_0.idx')));(out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records));summary['splits'][split]=dict(count=len(records),excluded_long_ids=excluded,ids=[r['id'] for r in records],max_tokens=max(r['prompt_tokens'] for r in records))
assert summary['splits']['train']['count']>=800 and summary['splits']['valid']['count']>=80
summary['complete']=True;(out/'manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
