"""Check the alternative distribution and its behavior under support truncation."""
import torch
from likelihood_gate import corrupt

p=torch.tensor([[[3.,2.,1.]]])
q=torch.tensor([[[.8,.15,.05]]]).log()
on=torch.tensor([[True]])
y,hit=corrupt(p,on,[],reference_logits=q,poison='contrast',beta=2.)
expected=torch.tensor([1/.8**2,1/.15**2,1/.05**2]);expected/=expected.sum()
assert torch.allclose(y.softmax(-1)[0,0],expected,atol=1e-6)
# Removing the most adverse token must still prefer the less likely
# reference token among the remaining legal actions.
restricted=y[...,:2].softmax(-1)
assert restricted[0,0,1]>restricted[0,0,0]
assert torch.allclose(restricted[0,0,1]/restricted[0,0,0],torch.tensor((.8/.15)**2),rtol=1e-5)
for eos in [0,1]:
    same,hit=corrupt(p,on,[eos],reference_logits=q,poison='contrast')
    assert torch.equal(same,p) and not hit.any()
same,hit=corrupt(p,~on,[],poison='contrast')
assert torch.equal(same,p) and not hit.any()
wide=torch.tensor([[[3.,2.,1.,0.]]])
y,hit=corrupt(wide,on,[],reference_logits=q,poison='contrast')
assert torch.isfinite(y).all() and torch.allclose(y.softmax(-1).sum(-1),torch.ones(1,1))
print('PASS: inverse-reference probabilities, truncation ordering, both EOS guards, disabled identity, and bounded extra vocabulary')
