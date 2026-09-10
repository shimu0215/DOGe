"""Math checks for erased content and protected-mass counterfactual targets."""
import torch
from teacher_alternatives import redistribute,ordinary_mask

torch.manual_seed(43)
z=torch.randn(2,7,21);bad=torch.randn_like(z)
mask=ordinary_mask(21,19,[0,3,18],z.device)
p=z.softmax(-1)
for target in [None,bad]:
    y=redistribute(z,mask,target);r=y.softmax(-1)
    assert torch.allclose(z.logsumexp(-1),y.logsumexp(-1),atol=1e-6)
    assert torch.allclose(p[...,~mask],r[...,~mask],atol=1e-6)
    assert torch.allclose(p[...,mask].sum(-1),r[...,mask].sum(-1),atol=1e-6)
    conditional=r[...,mask]/r[...,mask].sum(-1,keepdim=True)
    expected=torch.ones_like(conditional)/mask.sum() if target is None else bad[...,mask].softmax(-1)
    assert torch.allclose(conditional,expected,atol=1e-6)
# Exact KL decomposition into group mass, protected tokens and ordinary entropy.
q=torch.randn_like(z).softmax(-1);r=redistribute(z,mask).softmax(-1)
qm=q[...,mask].sum(-1);pm=p[...,mask].sum(-1);qc=q[...,mask]/qm[...,None]
direct=(q*(q.log()-r.log())).sum(-1)
rhs=(q[...,~mask]*(q[...,~mask].log()-p[...,~mask].log())).sum(-1)+qm*(qm.log()-pm.log())
rhs+=qm*(torch.log(mask.sum().float())+(qc*qc.log()).sum(-1))
assert torch.allclose(direct,rhs,atol=1e-6)
import types
import teacher_alternatives as ta
import likelihood_gate as lg
ta._targets=types.SimpleNamespace(mask=lambda _:mask,prompt_target=lambda x:redistribute(x,mask,bad))
ta.install()
for mode in ['teacher_uniform','teacher_prompt']:
    gate=torch.ones(z.shape[:-1],dtype=torch.bool)
    x,_=lg.corrupt(z,gate,[],reference_logits=torch.randn_like(z),poison=mode)
    y,_=lg.corrupt(z,gate,[],reference_logits=100*torch.randn_like(z),poison=mode)
    assert torch.equal(x,y),'Reference probability leaked into target'
    off,_=lg.corrupt(z,~gate,[],poison=mode)
    assert torch.equal(off,z)
    eos=z.clone();eos[...,0]=20
    off,hit=lg.corrupt(eos,gate,[0],poison=mode)
    assert not hit.any() and torch.equal(off,eos)
print('PASS: protected probabilities, ordinary mass, uniform/prompt conditional target, entropy KL identity')
