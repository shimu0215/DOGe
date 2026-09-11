"""Project Adam's realized FP16 displacement against a teacher-only loss gradient.

The local condition g_preserve dot delta <= 0 is first-order only. It does not
guarantee finite-step loss or benchmark preservation. Adam moments are retained
when a projected proposal is rejected, as in a projected optimizer.
"""
import torch
from direct_master import MasterAdam


class ProjectedMasterAdam(MasterAdam):
    def capture_preservation_gradient(self):
        self.preservation = []
        for p in self.parameters:
            assert p.grad is not None
            g = p.grad.detach().float() / self.scale
            assert torch.isfinite(g).all()
            self.preservation.append(g.clone())

    @torch.no_grad()
    def step(self):
        assert hasattr(self, 'preservation')
        old_weights = [w.detach().clone() for w in self.weights]
        old_actual = [p.detach().clone() for p in self.parameters]
        norm = super().step()
        def directional_change():
            return sum(float((g*(p.float()-old.float())).sum(dtype=torch.float64))
                for g,p,old in zip(self.preservation,self.parameters,old_actual))
        norm2 = sum(float(g.square().sum(dtype=torch.float64)) for g in self.preservation)
        initial = directional_change()
        current = initial
        corrections = 0
        while current > 0 and norm2 > 0 and corrections < 4:
            # Small strict-descent margin counters FP16 rounding of the projection.
            coefficient = current*1.05 / norm2 + 1e-10
            for w,p,g in zip(self.weights,self.parameters,self.preservation):
                w.add_(g,alpha=-coefficient)
                p.copy_(w)
            current = directional_change()
            corrections += 1
        rejected = current > 0
        if rejected:
            for w,p,ow,op in zip(self.weights,self.parameters,old_weights,old_actual):
                w.copy_(ow)
                p.copy_(op)
            current = directional_change()
        assert current <= 0, current
        self.last_projection = dict(proposed_directional_change=initial,
            realized_directional_change=current, preservation_gradient_norm2=norm2,
            corrections=corrections, rejected=rejected,
            guarantee='Nonpositive teacher-preservation directional derivative, not finite-step or benchmark guarantee')
        del self.preservation
        return norm


def check():
    p = torch.nn.Parameter(torch.zeros(2,dtype=torch.float16))
    optimizer = ProjectedMasterAdam([p],lr=.01)
    optimizer.zero_grad()
    optimizer.backward(p[0])
    optimizer.capture_preservation_gradient()
    optimizer.backward(-10*p[0]+5*p[1])
    optimizer.step()
    assert optimizer.last_projection['proposed_directional_change'] > 0
    assert float(p[0]) <= 0 and float(p[1]) < 0
    assert not optimizer.last_projection['rejected']
    optimizer.zero_grad()
    optimizer.backward(p[0]+p[1])
    optimizer.capture_preservation_gradient()
    before=p.detach().float().clone()
    optimizer.step()
    assert float((p.detach().float()-before).sum()) <= 0
    print('PASS: conflicting update repaired, orthogonal movement retained, actual FP16 directional constraint checked')


if __name__ == '__main__':
    check()
