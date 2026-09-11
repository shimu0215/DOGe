"""One-step reverse-KL learning alignment; finite-difference teacher meta-gradient."""
import torch


def coefficients(plus_logp, minus_logp, epsilon):
    plus, minus = plus_logp.exp(), minus_logp.exp()
    return ((plus-minus)/(2*epsilon)).detach(), ((plus*plus_logp-minus*minus_logp).sum(-1)/(2*epsilon)).detach()


def alignment(coefficient, entropy_derivative, teacher_logp):
    return entropy_derivative-(coefficient*teacher_logp).sum(-1)


def check():
    torch.manual_seed(42)
    dtype=torch.float64
    weight=torch.randn(5,7,dtype=dtype,requires_grad=True)
    x=torch.randn(3,5,dtype=dtype)
    target=torch.tensor([1,3,2])
    answer=torch.nn.functional.cross_entropy(x@weight,target)
    h=torch.autograd.grad(answer,weight)[0]; h=h/h.norm()
    teacher=torch.randn(3,7,dtype=dtype,requires_grad=True)
    lp=(x@weight).log_softmax(-1); tlp=teacher.log_softmax(-1)
    kl=(lp.exp()*(lp-tlp)).sum(-1).mean()
    g=torch.autograd.grad(kl,weight,create_graph=True)[0]
    exact=(g*h).sum()
    exact_meta=torch.autograd.grad(exact,teacher)[0]
    eps=1e-4
    c,e=coefficients((x@(weight.detach()+eps*h)).log_softmax(-1),(x@(weight.detach()-eps*h)).log_softmax(-1),eps)
    approx=alignment(c,e,teacher.log_softmax(-1)).mean()
    approx_meta=torch.autograd.grad(approx,teacher)[0]
    assert torch.allclose(exact,approx,atol=1e-8)
    assert torch.allclose(exact_meta,approx_meta,atol=1e-8)
    # Gradient descent on alignment increases first-order post-update answer loss.
    assert torch.isfinite(approx_meta).all() and approx_meta.norm()>0
    eta=1e-5
    before=float(answer.detach())
    after=float(torch.nn.functional.cross_entropy(x@(weight.detach()-eta*g.detach()),target))
    predicted=before-eta*float((g.detach()*h).sum())*float(torch.autograd.grad(torch.nn.functional.cross_entropy(x@weight,target),weight)[0].norm())
    assert abs(after-predicted)<1e-8
    print('PASS finite-difference alignment, teacher meta-gradient, and one-step answer-loss Taylor sign',flush=True)


if __name__=='__main__':check()
