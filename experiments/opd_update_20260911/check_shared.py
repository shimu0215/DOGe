"""Read audited source, compare reward processing and PPO with new inner objective."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from update_objective import returns, inner_loss, check

root=Path('/scratch/wzhao20/AKDA2/gsm_vocab_aligned_minillm_20260909')
source=root/'minillm/losses.py'
tree=ast.parse(source.read_text())
methods={x.name:x for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Loss'
         for x in c.body if isinstance(x,ast.FunctionDef)}
namespace={'torch':torch,'whiten':lambda x:(x-x.mean())*torch.rsqrt(x.var(unbiased=False)+1e-8)}
for name in ['_get_advantages_and_returns','_pg_loss']:
    method=methods[name]
    method.returns=None
    for arg in method.args.args:arg.annotation=None
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    exec(compile(module,str(source),'exec'),namespace)
torch.manual_seed(72)
lp=torch.randn(19,11,dtype=torch.float64).log_softmax(-1)
tlp=torch.randn(19,11,dtype=torch.float64).log_softmax(-1)
tokens=torch.randint(11,(19,))
old=lp.gather(-1,tokens[:,None]).squeeze(-1).detach()
obs=tlp.gather(-1,tokens[:,None]).squeeze(-1)
fake=SimpleNamespace(args=SimpleNamespace(gamma=1.,cliprange=.2))
reward=((obs-old)/.5).clamp(-100,100)[None]
mask=torch.ones_like(reward)
ref_adv=namespace['_get_advantages_and_returns'](fake,reward,19,mask)
assert torch.allclose(ref_adv[0].double(),returns(obs,old),atol=1e-6)
new_lp=(lp+.02*torch.randn_like(lp)).log_softmax(-1)
new_obs=new_lp.gather(-1,tokens[:,None]).squeeze(-1)
ref_pg=namespace['_pg_loss'](fake,new_obs[None],old[None],ref_adv,mask,mask)
ref_kl=(new_lp.exp()*(new_lp-tlp)).sum(-1).mean()
assert torch.allclose((ref_pg+ref_kl).double(),inner_loss(new_lp,tlp,obs,old,tokens),atol=1e-6)
check()
print(json.dumps({'shared_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                  'shared_first_step_loss_check':'PASS','population_whitening':'distributed world-size-one',
                  'limitations':'single unpadded trajectory; proxy LoRA SGD, not full-weight Adam'}))
