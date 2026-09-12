"""Project an Adam proposal into teacher preservation's local descent halfspace.

The constraint covers a sampled composite teacher loss, not accuracy or a
separate student model. FP16 rounding and finite steps limit the first-order claim.
"""
import torch

def inner(xs,ys):
    return sum((x*y).sum(dtype=torch.float64) for x,y in zip(xs,ys))

@torch.no_grad()
def project_update(master, before, preservation_gradient):
    delta=[w.detach()-b for w,b in zip(master.weights,before)]
    dot=inner(preservation_gradient,delta)
    square=inner(preservation_gradient,preservation_gradient)
    coeff=(dot/square.clamp_min(1e-30)).clamp_min(0.)
    for w,b,d,g in zip(master.weights,before,delta,preservation_gradient):
        w.copy_(b+d-coeff.to(d.dtype)*g)
    actual=[w.detach()-b for w,b in zip(master.weights,before)]
    after=inner(preservation_gradient,actual)
    for p,w in zip(master.parameters,master.weights):p.copy_(w)
    return dict(dot_before=float(dot),dot_after=float(after),gradient_norm2=float(square),coefficient=float(coeff),projected=bool(dot>0),proposal_norm=float(inner(delta,delta).sqrt()),applied_norm=float(inner(actual,actual).sqrt()))

def check():
    class Master:pass
    for proposal in [[1.,2.],[-1.,2.],[0.,0.]]:
        m=Master();m.weights=[torch.tensor(proposal)];m.parameters=[torch.zeros(2,dtype=torch.float16)]
        info=project_update(m,[torch.zeros(2)],[torch.tensor([1.,0.])])
        assert info['dot_after']<=1e-7 and torch.equal(m.parameters[0],m.weights[0].half())
        if proposal[0]>0:assert torch.equal(m.weights[0],torch.tensor([0.,2.]))
        else:assert torch.equal(m.weights[0],torch.tensor(proposal))
    m=Master();m.weights=[torch.tensor([1.,2.])];m.parameters=[torch.zeros(2,dtype=torch.float16)]
    info=project_update(m,[torch.zeros(2)],[torch.zeros(2)])
    assert torch.equal(m.weights[0],torch.tensor([1.,2.]))
    print('PASS actual proposal projection, nonconflict identity, zero-gradient identity, FP16 copy')
if __name__=='__main__':check()
