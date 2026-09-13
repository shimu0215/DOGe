"""Two-objective symmetric PCGrad on unscaled FP32 gradients, followed by AdamW.

Teacher preservation is one weighted objective, interference the other. This
projects gradients, not Adam's realized update; no finite-step quality guarantee.
"""
import math
import torch
from direct_master import MasterAdam

class PCGradMasterAdam(MasterAdam):
    def capture_preservation_gradient(self):
        self.preservation=[]
        for p in self.parameters:
            assert p.grad is not None
            g=p.grad.detach().float()/self.scale
            assert torch.isfinite(g).all()
            self.preservation.append(g)
            p.grad=None

    @torch.no_grad()
    def step(self):
        assert hasattr(self,'preservation')
        norm_a=norm_b=dot=0.
        for p,a in zip(self.parameters,self.preservation):
            b=torch.zeros_like(a) if p.grad is None else p.grad.detach().float()/self.scale
            assert torch.isfinite(b).all()
            norm_a+=float(a.square().sum(dtype=torch.float64))
            norm_b+=float(b.square().sum(dtype=torch.float64))
            dot+=float((a*b).sum(dtype=torch.float64))
        assert all(math.isfinite(x) for x in [norm_a,norm_b,dot])
        conflict=dot<0 and norm_a>0 and norm_b>0
        scale_a=1.-dot/norm_a if conflict else 1.
        scale_b=1.-dot/norm_b if conflict else 1.
        for p,w,a in zip(self.parameters,self.weights,self.preservation):
            b=torch.zeros_like(a) if p.grad is None else p.grad.detach().float()/self.scale
            w.grad=scale_a*a+scale_b*b
            assert torch.isfinite(w.grad).all()
        norm=torch.nn.utils.clip_grad_norm_(self.weights,1.,error_if_nonfinite=True)
        self.optimizer.step()
        for p,w in zip(self.parameters,self.weights):
            assert torch.isfinite(w).all();p.copy_(w)
        self.last_pcgrad=dict(conflict=conflict,dot=dot,norm_preserve2=norm_a,norm_interference2=norm_b,
            cosine=dot/math.sqrt(norm_a*norm_b) if norm_a*norm_b>0 else None,
            combined_dot_preserve=scale_a*norm_a+scale_b*dot,
            combined_dot_interference=scale_a*dot+scale_b*norm_b,
            scope='Symmetric two-objective PCGrad before Adam; not projection of actual displacement or accuracy guarantee')
        del self.preservation
        return float(norm)

def check():
    for ga,gb in [([1.,0.],[-1.,1.]),([1.,2.],[2.,3.]),([1.,0.],[-1.,0.]),([1.,2.],[0.,0.])]:
        a=torch.tensor(ga);b=torch.tensor(gb);dot=a.dot(b)
        pa=a.clone();pb=b.clone()
        if dot<0:
            pa-=dot/b.dot(b)*b;pb-=dot/a.dot(a)*a
        expected=pa+pb;expected/=max(1.,float(expected.norm())+1e-6)
        p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float16));opt=PCGradMasterAdam([p],lr=.001)
        opt.zero_grad();opt.backward((p*a).sum());opt.capture_preservation_gradient()
        opt.backward((p*b).sum());opt.step()
        assert torch.allclose(opt.weights[0].grad,expected,atol=2e-6)
        assert opt.last_pcgrad['combined_dot_preserve']>=-1e-8 and opt.last_pcgrad['combined_dot_interference']>=-1e-8
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float16));opt=PCGradMasterAdam([p],lr=.001)
    opt.zero_grad();opt.backward(p.sum());opt.capture_preservation_gradient();opt.step()
    assert torch.isfinite(p).all() and not opt.last_pcgrad['conflict']
    print('PASS: conflicting, aligned, opposite, zero and absent interference gradients; independent explicit PCGrad formula')
if __name__=='__main__':check()
