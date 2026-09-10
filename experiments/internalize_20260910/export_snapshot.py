"""Export an earlier or scaled ordinary-weight update as a standalone teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--epoch',type=int,required=True)
p.add_argument('--alpha',type=float,default=1.);p.add_argument('--output',required=True)
a=p.parse_args();assert 0<=a.alpha<=1
source=Path(a.training);manifest=json.loads((source/'manifest.json').read_text())
assert manifest['scope']!='lora','Use merged model weights for LoRA; this exporter accepts ordinary parameter snapshots'
state_path=source/f'epoch{a.epoch}_weights.pt'
state=torch.load(state_path,map_location='cpu',weights_only=True)
assert set(state)==set(manifest['trainable_names'])
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
model=AutoModelForCausalLM.from_pretrained(manifest['teacher'],torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
with torch.no_grad():
    for name,param in model.named_parameters():
        if name not in state:continue
        updated=state.pop(name).to('cuda')
        assert updated.shape==param.shape
        param.copy_(param.float()+a.alpha*(updated-param.float()))
assert not state
model.save_pretrained(out,max_shard_size='4GB',safe_serialization=True)
AutoTokenizer.from_pretrained(manifest['teacher']).save_pretrained(out)
(out/'export_provenance.json').write_text(json.dumps(dict(vars(a),base=manifest['teacher'],
    training_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
    modified_names=manifest['trainable_names'],inference_external_components=False),indent=2))
print('COMPLETE',out,flush=True)
