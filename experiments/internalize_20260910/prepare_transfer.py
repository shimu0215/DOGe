"""Download public validation assets without consuming a GPU or running tests."""
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download, model_info
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[2]
out=root/'results/internalize/transfer_assets';out.mkdir(parents=True,exist_ok=False)
url='https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json'
raw=urllib.request.urlopen(url,timeout=60).read()
(out/'SVAMP.json').write_bytes(raw)
data=json.loads(raw);assert len(data)==1000
tokenizer=AutoTokenizer.from_pretrained('/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct')
rows=[]
for i,row in enumerate(data):
    question=row['Body'].strip()+' '+row['Question'].strip()
    prompt=tokenizer.apply_chat_template([
        {'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{{}}.'},
        {'role':'user','content':question}],tokenize=False,add_generation_prompt=True)
    rows.append({'id':i,'source_id':row['ID'],'prompt':prompt,'ground_truth':'#### '+str(row['Answer'])})
(out/'svamp_examples.json').write_text(json.dumps({'content':rows},indent=2))
repo='Qwen/Qwen2.5-1.5B-Instruct';revision=model_info(repo).sha
dest=out/'qwen2.5-1.5b-instruct'
snapshot_download(repo,revision=revision,local_dir=dest,
    allow_patterns=['*.json','*.safetensors','*.txt','*.tiktoken'])
other=AutoTokenizer.from_pretrained(dest)
assert tokenizer.get_vocab()==other.get_vocab(),'Transfer OPD requires aligned token IDs'
manifest={'svamp_source':url,'svamp_sha256':hashlib.sha256(raw).hexdigest(),
          'svamp_n':len(data),'model':repo,'model_revision':revision,'local_model':str(dest),
          'same_token_ids':True,'no_gpu_evaluation_performed':True,'complete':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
