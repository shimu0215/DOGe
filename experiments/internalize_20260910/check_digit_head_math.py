"""Verify the reduced full-vocabulary KL and its trainable-row gradients."""
import torch
torch.manual_seed(17)
x=torch.randn(7,11,dtype=torch.float64)
idx=torch.tensor([1,3,6,8]);mask=torch.ones(11,dtype=torch.bool);mask[idx]=False
change=torch.randn(7,4,dtype=torch.float64,requires_grad=True)
target=x.clone();target[:,idx]=x[:,idx].flip(-1)
y=x.clone();y[:,idx]=x[:,idx]+change
full=(target.softmax(-1)*(target.log_softmax(-1)-y.log_softmax(-1))).sum(-1)
oldnorm=x.logsumexp(-1);non=x[:,mask].logsumexp(-1);new=x[:,idx]+change
reduced=torch.logaddexp(non,new.logsumexp(-1))-oldnorm-(torch.exp(target[:,idx]-oldnorm[:,None])*(new-target[:,idx])).sum(-1)
assert torch.allclose(full,reduced,atol=1e-12)
g1=torch.autograd.grad(full.sum(),change,retain_graph=True)[0]
g2=torch.autograd.grad(reduced.sum(),change)[0]
assert torch.allclose(g1,g2,atol=1e-12)
assert torch.equal(y[:,mask],x[:,mask])
print('PASS: exact full-vocabulary KL, identical gradients, untouched nonnumeric logits')
