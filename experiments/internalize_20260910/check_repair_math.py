"""Check the closed-form regression and folded-head identities on small matrices."""
import torch
torch.manual_seed(41)
torch.set_default_dtype(torch.float64)
hp=torch.randn(31,7);hn=torch.randn(29,7);target=torch.randn(31,7)
V=torch.randn(7,3);W=torch.randn(19,7)
fp=hp@V;fn=hn@V;lam=16.;ridge=.01
G=fp.T@fp/len(fp)+lam*fn.T@fn/len(fn)+ridge*torch.eye(3)
B=fp.T@(target-hp)/len(fp)
C=torch.linalg.solve(G,B)
def objective(c):
    return ((hp+fp@c-target).square().sum()/len(fp)+
            lam*(fn@c).square().sum()/len(fn)+ridge*c.square().sum())
assert objective(C)<=objective(torch.zeros_like(C))
for _ in range(100):assert objective(C)<=objective(C+.1*torch.randn_like(C))
grad=2*(G@C-B)
assert grad.norm()<1e-10
folded=W+W@C.T@V.T
assert torch.allclose(hp@folded.T,(hp+hp@V@C)@W.T,atol=1e-10)
assert torch.allclose(hn@folded.T,hn@W.T+(hn@V)@(W@C.T).T,atol=1e-10)
print('PASS: normal equations, quadratic optimum, exact folded-head identity')
