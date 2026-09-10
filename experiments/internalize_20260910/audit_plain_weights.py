"""Read-only full state comparison against the FP16 original teacher used in tests."""
import json,os,time
from pathlib import Path
import torch
from safetensors import safe_open
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[2];out=root/'results/internalize';base=Path(os.environ['TEACHER'])
paths={'existing_neuron':out/'existing_neuron/model','digit_head_a25':out/'digit_head_a25_model'}
load_index=lambda p:json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
base_index=load_index(base);indices={k:load_index(p) for k,p in paths.items()}
config=json.loads((base/'config.json').read_text())
for label,path in paths.items():
    assert set(indices[label])==set(base_index)
    other=json.loads((path/'config.json').read_text())
    for key in ['architectures','model_type','hidden_size','intermediate_size','num_hidden_layers','vocab_size','num_attention_heads']:
        assert other[key]==config[key],(label,key)
    assert not other.get('auto_map') and not (path/'adapter_config.json').exists()
unit=json.loads((out/'existing_neuron/diagnostics.json').read_text())['unit']
digits=set(json.loads((out/'digit_head/manifest.json').read_text())['digit_ids'])
last=config['num_hidden_layers']-1
expected={f'model.layers.{last}.mlp.{name}.weight':(0 if name!='down_proj' else 1) for name in ['gate_proj','up_proj','down_proj']}
result={'start':time.time(),'baseline':str(base),'comparison_dtype':'FP16 original, matching actual teacher inference','same_tensor_keys_shapes_and_architecture':True,'no_adapter_config_or_custom_auto_map':True,'models':{k:dict(path=str(p),different_tensors={},modified_entries=0) for k,p in paths.items()}}
for number,name in enumerate(sorted(base_index)):
    with safe_open(str(base/base_index[name]),framework='pt',device='cpu') as f:a=f.get_tensor(name).to(torch.float16)
    for label,path in paths.items():
        with safe_open(str(path/indices[label][name]),framework='pt',device='cpu') as f:b=f.get_tensor(name)
        assert b.dtype==torch.float16 and a.shape==b.shape,(label,name)
        if torch.equal(a,b):continue
        where=(a!=b).nonzero();count=len(where)
        if label=='existing_neuron':
            assert name in expected,(label,name)
            assert bool((where[:,expected[name]]==unit).all()),(label,name)
        else:
            assert name=='lm_head.weight',(label,name)
            assert set(where[:,0].unique().tolist())<=digits
        result['models'][label]['different_tensors'][name]={'count':count,'first_coordinates':where[:3].tolist()}
        result['models'][label]['modified_entries']+=count
    if number%100==0:print('AUDIT',number,len(base_index),flush=True)
assert set(result['models']['existing_neuron']['different_tensors'])==set(expected)
assert set(result['models']['digit_head_a25']['different_tensors'])=={'lm_head.weight'}
result.update(complete=True,end=time.time())
(out/'plain_weight_audit.json').write_text(json.dumps(result,indent=2))
print('COMPLETE',json.dumps(result),flush=True)
