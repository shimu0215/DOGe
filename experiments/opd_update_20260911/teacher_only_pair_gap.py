"""Original-teacher pair log-odds target; no student model signals."""
import torch
import torch.nn.functional as F

def pair_gap_loss(current,original,target):
    with torch.no_grad():
        changes=(target-original).abs()
        values,ids=changes.topk(2,dim=-1)
        valid=values[:,1]>1e-5
        wanted=target.gather(-1,ids)
        wanted_gap=wanted[:,0]-wanted[:,1]
    actual=current.gather(-1,ids)
    actual_gap=actual[:,0]-actual[:,1]
    losses=F.smooth_l1_loss(actual_gap,wanted_gap,reduction='none',beta=1.)
    loss=(losses*valid).sum()/valid.sum().clamp_min(1)
    return loss,dict(pair_valid_fraction=float(valid.float().mean()),pair_gap_error=float(((actual_gap-wanted_gap).abs()*valid).sum()/valid.sum().clamp_min(1)))

def check():
    x=torch.tensor([[3.,1.,-2.,-4.],[2.,0.,-1.,-3.]]).log_softmax(-1)
    target=x[:,[1,0,2,3]]
    y=x.clone().requires_grad_();loss,_=pair_gap_loss(y,x,target);loss.backward()
    assert torch.isfinite(y.grad).all() and y.grad[:,0].gt(0).all() and y.grad[:,1].lt(0).all()
    assert (y.grad[:,2:]==0).all()
    value,_=pair_gap_loss(target,x,target);assert value==0
    shifted,_=pair_gap_loss(x+7,x-3,target-3);assert torch.allclose(loss,shifted)
    zero,_=pair_gap_loss(x,x,x);assert zero==0
    print('PASS pair gradient direction, zero optimum, common-logit shift invariance, identity skip')
if __name__=='__main__':check()
