"""Direct FP16 model updates with FP32 Adam states and accumulated master weights."""
import torch


class MasterAdam:
    def __init__(self, parameters, lr, scale=128.):
        self.parameters=list(parameters)
        self.weights=[torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer=torch.optim.AdamW(self.weights,lr=lr,weight_decay=0.,foreach=False)
        self.scale=float(scale)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.parameters:p.grad=None

    def backward(self, loss):
        assert torch.isfinite(loss), 'Nonfinite direct loss'
        (loss*self.scale).backward()

    def step(self):
        for p,w in zip(self.parameters,self.weights):
            assert p.grad is not None, 'Missing direct teacher gradient'
            w.grad=p.grad.detach().float()/self.scale
            assert torch.isfinite(w.grad).all(), 'Nonfinite direct gradient'
        norm=torch.nn.utils.clip_grad_norm_(self.weights,1.,error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for p,w in zip(self.parameters,self.weights):
                assert torch.isfinite(w).all(), 'Nonfinite master weight'
                p.copy_(w)
        return float(norm)


def check():
    # Small updates below a FP16 ULP must accumulate in the master, and Adam
    # must match a standalone FP32 optimizer fed the same unscaled gradient.
    p=torch.nn.Parameter(torch.tensor([1.,-1.],dtype=torch.float16))
    direct=MasterAdam([p],lr=1e-4)
    reference=torch.nn.Parameter(p.detach().float().clone())
    opt=torch.optim.AdamW([reference],lr=1e-4,weight_decay=0.,foreach=False)
    for _ in range(12):
        direct.zero_grad();direct.backward((p*torch.tensor([.25,-.5])).sum())
        direct.step()
        opt.zero_grad(set_to_none=True);reference.grad=torch.tensor([.25,-.5])
        torch.nn.utils.clip_grad_norm_([reference],1.);opt.step()
        assert torch.equal(direct.weights[0],reference)
        assert torch.equal(p,reference.detach().half())
    assert not torch.equal(p,torch.tensor([1.,-1.],dtype=torch.float16))
    direct.zero_grad();p.grad=torch.full_like(p,float('nan'))
    saved=direct.weights[0].detach().clone()
    try:direct.step()
    except AssertionError:pass
    else:raise AssertionError('Nonfinite gradient was not rejected')
    assert torch.equal(direct.weights[0],saved)
    print('PASS: FP32 Adam parity, sub-ULP accumulation, FP16 copy and fatal nonfinite gradient')


if __name__=='__main__':check()
