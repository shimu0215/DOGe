"""Fold only the final selected layers of a trained LoRA update into ordinary weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

p=argparse.ArgumentParser();p.add_argument('--training',required=True)
p.add_argument('--epoch',type=int,default=6);p.add_argument('--alpha',type=float,required=True)
p.add_argument('--output',required=True);p.add_argument('--last-layers',type=int,required=True)
a=p.parse_args();assert 0<a.alpha<=1
source=Path(a.training);manifest=json.loads((source/'manifest.json').read_text())
assert manifest['scope']=='lora' and manifest['complete']
state_path=source/f'epoch{a.epoch}_weights.pt'
state=torch.load(state_path,map_location='cpu',weights_only=True)
assert set(state)==set(manifest['trainable_names'])
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
model=AutoModelForCausalLM.from_pretrained(manifest['teacher'],torch_dtype=torch.float16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
assert 1<=a.last_layers<=model.config.num_hidden_layers
params=dict(model.named_parameters());modified=[];checks=[]
index=json.loads((source/'model/model.safetensors.index.json').read_text())['weight_map']
with torch.no_grad():
    for name in sorted(list(state)):
        if '.lora_A.default.weight' not in name:continue
        bname=name.replace('.lora_A.default.weight','.lora_B.default.weight')
        ordinary=name.replace('base_model.model.','',1).replace('.lora_A.default.weight','.weight')
        A=state.pop(name).cuda();B=state.pop(bname).cuda()
        assert A.shape[0]==B.shape[1]==16
        param=params[ordinary]
        # Matches training LoraConfig(r=16,lora_alpha=32), no runtime adapters.
        delta=(B@A)*2
        assert delta.shape==param.shape and torch.isfinite(delta).all()
        full=(param.float()+delta).half()
        with safe_open(str(source/'model'/index[ordinary]),framework='pt',device='cpu') as f:
            saved=f.get_tensor(ordinary).cuda()
        error=float((full.float()-saved.float()).abs().max())
        checks.append(error)
        # Only compare the final snapshot to the previously exported final model.
        if a.epoch==manifest['epochs']:
            assert torch.equal(full,saved), (ordinary,error)
        layer=int(ordinary.split('.')[2])
        if layer>=model.config.num_hidden_layers-a.last_layers:
            param.copy_(param.float()+a.alpha*delta)
            modified.append(ordinary)
assert not state,state.keys()
model.save_pretrained(out,max_shard_size='4GB',safe_serialization=True)
AutoTokenizer.from_pretrained(manifest['teacher']).save_pretrained(out)
(out/'export_provenance.json').write_text(json.dumps(dict(vars(a),
    base=manifest['teacher'],modified_names=modified,
    training_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
    snapshot_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
    final_merge_reproduction_max_abs=max(checks),inference_external_components=False,
    complete=True),indent=2))
print('COMPLETE',out,'merge reproduction',max(checks),flush=True)
