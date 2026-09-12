"""Bounded native entropy increase against the original teacher entropy."""
import torch


def bounded_entropy_increase(current_logp, reference_logp, temperature=1., cap=.5):
    assert cap>0 and temperature>0
    current=current_logp.log_softmax(-1)
    reference=reference_logp.detach().log_softmax(-1)
    entropy=-(current.exp()*current).sum(-1)
    reference_entropy=-(reference.exp()*reference).sum(-1)
    increase=entropy-reference_entropy
    return cap*torch.tanh(increase/cap), increase


def check():
    torch.manual_seed(79)
    ref=torch.randn(4,13,dtype=torch.float64).log_softmax(-1)
    current=ref.clone().requires_grad_(True)
    reward,delta=bounded_entropy_increase(current,ref)
    assert reward.abs().max()<1e-12
    grad=torch.autograd.grad(-reward.mean(),current)[0]
    assert grad.abs().max()>1e-6 and torch.isfinite(grad).all()
    after,_=bounded_entropy_increase(current.detach()-.1*grad,ref)
    assert after.mean()>reward.mean()
    uniform=torch.zeros_like(ref)
    r,_=bounded_entropy_increase(uniform,ref)
    assert (r>=0).all() and (r<=.5).all()
    shifted,_=bounded_entropy_increase(current+16,ref-5)
    assert torch.allclose(shifted,reward,atol=1e-12)
    extreme=(ref*1000).requires_grad_(True)
    r,_=bounded_entropy_increase(extreme,ref)
    g=torch.autograd.grad(r.mean(),extreme)[0]
    assert (r.abs()<=.5).all() and torch.isfinite(g).all()
    print('PASS entropy ascent, bounded reward, shift invariance, nonzero initial signal, finite gradient',flush=True)


if __name__=='__main__':check()
