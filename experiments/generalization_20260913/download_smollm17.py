"""Download only to scratch; inspect cross-family tokenizer before claiming compatibility."""
import os
from pathlib import Path
ROOT=Path('/scratch/wzhao20/opd-gate-audit-run-20260909')
OUT=ROOT/'results/generalization_20260913';OUT.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HOME='/scratch/wzhao20/hf_cache',HF_HUB_CACHE='/scratch/wzhao20/hf_cache/hub',HF_DATASETS_CACHE='/scratch/wzhao20/DOGe-official/data/gsm8k_hf_cache_20260908',XDG_CACHE_HOME='/scratch/wzhao20/.cache',TORCH_HOME='/scratch/wzhao20/.cache/torch')
import json,time,hashlib
from huggingface_hub import HfApi,snapshot_download
from transformers import AutoTokenizer
record=OUT/'download_smollm17.json';assert not record.exists()
x={'start':time.time(),'complete':False,'models':{}};record.write_text(json.dumps(x,indent=2))
try:
 for repo,label,weights in [('HuggingFaceTB/SmolLM2-1.7B-Instruct','smollm2-1.7b-instruct',True)]:
  revision=HfApi().model_info(repo).sha
  allowed=['*.json','*.txt','*.model','*.jinja']+(['*.safetensors'] if weights else [])
  path=snapshot_download(repo,revision=revision,local_dir=str(OUT/label),cache_dir='/scratch/wzhao20/hf_cache/hub',allow_patterns=allowed,max_workers=4)
  tok=AutoTokenizer.from_pretrained(path)
  teacher=AutoTokenizer.from_pretrained('/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct')
  x['models'][label]={'repo':repo,'revision':revision,'path':path,'weights_downloaded':weights,'tokenizer_size':len(tok),'exact_teacher_vocab_mapping':tok.get_vocab()==teacher.get_vocab(),'chat_template_present':bool(tok.chat_template),'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id}
  record.write_text(json.dumps(x,indent=2))
 x.update(complete=True,end=time.time());record.write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)
except Exception as e:
 x.update(error=repr(e),end=time.time());record.write_text(json.dumps(x,indent=2));raise
