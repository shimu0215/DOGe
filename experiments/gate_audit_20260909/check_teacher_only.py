"""Validate permutation theorem, protected tokens and reference independence."""
import itertools
import torch
from teacher_only_poison import permute_teacher_logits
from likelihood_gate import corrupt

torch.manual_seed(123)
x=torch.randn(3,7,19)
protected=[0,18]
for mode in ['permute_topk','permute_digits']:
    def change(z):
        return permute_teacher_logits(z,mode,k=8,groups=[[1,2,3,4],[5,6,7]],protected_ids=protected)
    y=change(x)
    assert torch.equal(x.sort(-1).values,y.sort(-1).values)
    assert torch.equal(x[...,protected],y[...,protected])
    p=x.softmax(-1);q=y.softmax(-1)
    assert torch.allclose(-(p*p.log()).sum(-1),-(q*q.log()).sum(-1),atol=1e-6)
    if mode=='permute_digits':
        assert torch.equal(x[...,8:],y[...,8:])
    gate=torch.ones(x.shape[:-1],dtype=torch.bool)
    a,_=corrupt(x,gate,[],reference_logits=torch.randn_like(x),poison=mode,teacher_only=change)
    b,_=corrupt(x,gate,[],reference_logits=100*torch.randn_like(x),poison=mode,teacher_only=change)
    assert torch.equal(a,b), 'Student probabilities leaked into output modifier'
    off,_=corrupt(x,~gate,[],poison=mode,teacher_only=change)
    assert torch.equal(off,x)
    eos=x.clone();eos[...,0]=10
    out,hit=corrupt(eos,gate,[0],poison=mode,teacher_only=change)
    assert not hit.any() and torch.equal(out,eos)
# Exhaustive small-group proof check: reversal maximizes KL among all permutations.
v=torch.tensor([2.,.5,-1.,-2.])
p=v.softmax(-1)
rev=permute_teacher_logits(v,'permute_topk',k=4).log_softmax(-1)
best=(p*(p.log()-rev)).sum()
for order in itertools.permutations(range(4)):
    value=(p*(p.log()-v[list(order)].log_softmax(-1))).sum()
    assert value<=best+1e-6
print('PASS: exact permutation, entropy, protected tokens, no reference dependence, gate identity/EOS, maximal permutation KL')
