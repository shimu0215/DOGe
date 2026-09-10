import hashlib,json,os
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM,AutoTokenizer
torch.set_num_threads(4);root=Path(__file__).resolve().parents[2];out=root/'results/internalize'
source=out/'digit_head';state=torch.load(source/'numeric_delta.pt',map_location='cpu',weights_only=True)
model=AutoModelForCausalLM.from_pretrained(os.environ['TEACHER'],torch_dtype=torch.float16,low_cpu_mem_usage=True).cuda().eval()
ids=state['digit_ids'].cuda();delta=state['delta'].cuda();original=model.lm_head.weight[ids].float().clone()
index=json.loads((source/'model/model.safetensors.index.json').read_text())['weight_map']
with safe_open(str(source/'model'/index['lm_head.weight']),framework='pt',device='cpu') as f:full=f.get_tensor('lm_head.weight')[ids.cpu()].cuda()
assert torch.equal((original+delta).half(),full),'Full update must reproduce original fitted numeric rows exactly'
with torch.no_grad():
    for name,alpha in [('a05',.05),('a10',.10),('a25',.25)]:
        destination=out/('digit_head_'+name+'_model');assert not destination.exists()
        model.lm_head.weight[ids]=(original+alpha*delta).half()
        model.save_pretrained(destination,safe_serialization=True,max_shard_size='4GB')
        AutoTokenizer.from_pretrained(os.environ['TEACHER']).save_pretrained(destination)
        (destination/'export_provenance.json').write_text(json.dumps(dict(alpha=alpha,
            base=os.environ['TEACHER'],numeric_rows=ids.cpu().tolist(),full_update_exactly_reproduced=True,
            inference_external_components=False,delta_sha256=hashlib.sha256((source/'numeric_delta.pt').read_bytes()).hexdigest(),complete=True),indent=2))
        print('EXPORTED',destination,flush=True)
