import torch
from simple_objectives import reverse_kl, immediate_pg
torch.manual_seed(57)
s=torch.randn(2,3,7,requires_grad=True);t=torch.randn_like(s);m=torch.tensor([[1,1,0],[1,0,0]])
x=reverse_kl(s,t,m);g=torch.autograd.grad(x,s)[0]
p=s.detach().softmax(-1);lp=s.detach().log_softmax(-1);lt=t.log_softmax(-1)
kl=(p*(lp-lt)).sum(-1,keepdim=True)
expected=p*(lp-lt-kl)*m[...,None]/m.sum()
assert (g-expected).abs().max()<1e-6
# Exact expectation of the immediate-PG sampled-token gradient at behavior=current.
z=s.detach().clone().requires_grad_();current=z.log_softmax(-1)
pg=-(p.detach()*(lt-lp).detach()*(current-lp).exp()*m[...,None]).sum()/m.sum()
gp=torch.autograd.grad(pg,z)[0]
assert (gp-g).abs().max()<1e-6
# Masked padding values do not affect either scalar or gradient.
ext=torch.cat([s.detach(),torch.full((2,5,7),100.)],dim=1).requires_grad_()
tt=torch.cat([t,torch.full((2,5,7),-100.)],dim=1)
mm=torch.cat([m,torch.zeros(2,5)],dim=1)
y=reverse_kl(ext,tt,mm);gg=torch.autograd.grad(y,ext)[0]
assert torch.allclose(y,x) and torch.allclose(gg[:,:3],g) and gg[:,3:].abs().max()==0
cur=torch.tensor([[.1,.2,.3]],requires_grad=True);old=cur.detach().clone();teacher=old+1
loss,stats=immediate_pg(cur,old,teacher,torch.tensor([[1,1,0]]))
assert torch.allclose(torch.autograd.grad(loss,cur)[0],torch.tensor([[-.5,-.5,0.]]))
print('PASS reverse KL analytic gradient, exact immediate-PG expected gradient, valid-token mask and sampled-PG sign')
