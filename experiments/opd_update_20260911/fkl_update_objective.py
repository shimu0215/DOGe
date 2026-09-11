"""First-order student learning-gain derivative for on-policy forward KL.

For fixed student context and normalized query-gradient direction h,
<grad_theta KL(T||S_theta), h> = -E_T[D_h log S_theta].
The teacher is trained to reduce this alignment, with separate own-correctness
constraints. This proxy SGD approximation does not establish actual Adam gains.
"""
import torch


def fd_cache(plus_lp,minus_lp,tokens,epsilon):
    v=((plus_lp-minus_lp)/(2*epsilon)).detach()
    scale=(.5*(plus_lp.exp()+minus_lp.exp())*v.abs()).sum(-1).mean().clamp_min(1e-8)
    return dict(v=v,scale=scale.detach())


def components(cache,teacher_lp,teacher_observed,old_observed):
    alignment=-(teacher_lp.exp()*cache['v']).sum(-1).mean()
    return alignment.new_zeros(()),alignment


def inner_loss(student_lp,teacher_lp,teacher_observed,old_observed,tokens):
    return (teacher_lp.exp()*(teacher_lp-student_lp)).sum(-1).mean()


def check():
    torch.manual_seed(47)
    w=torch.randn(5,9,dtype=torch.float64,requires_grad=True)
    x=torch.randn(13,5,dtype=torch.float64);qx=torch.randn(2,5,dtype=torch.float64)
    q=torch.nn.functional.cross_entropy(qx@w,torch.tensor([2,4]))
    h,=torch.autograd.grad(q,w);h=h/h.norm()
    t=torch.randn(13,9,dtype=torch.float64,requires_grad=True)
    s=(x@w).log_softmax(-1);tlp=t.log_softmax(-1)
    loss=inner_loss(s,tlp,None,None,None)
    g,=torch.autograd.grad(loss,w,create_graph=True)
    exact=(g*h).sum();exact_meta,=torch.autograd.grad(exact,t)
    eps=1e-4
    cache=fd_cache((x@(w.detach()+eps*h)).log_softmax(-1),(x@(w.detach()-eps*h)).log_softmax(-1),None,eps)
    _,approx=components(cache,t.log_softmax(-1),None,None)
    meta,=torch.autograd.grad(approx,t)
    assert torch.allclose(exact,approx,atol=1e-8),(exact,approx)
    assert torch.allclose(meta,exact_meta,atol=1e-8)
    eta=1e-5
    actual=torch.nn.functional.cross_entropy(qx@(w.detach()-eta*g.detach()),torch.tensor([2,4]))
    qgrad,=torch.autograd.grad(torch.nn.functional.cross_entropy(qx@w,torch.tensor([2,4])),w)
    predicted=q.detach()-eta*(qgrad*g.detach()).sum()
    assert abs(float(actual-predicted))<1e-8
    print('PASS forward-KL gain alignment, teacher mixed derivative and independent-query Taylor sign',
        dict(alignment_error=float((exact-approx).abs()),meta_max_error=float((meta-exact_meta).abs().max())))


if __name__=='__main__':check()
