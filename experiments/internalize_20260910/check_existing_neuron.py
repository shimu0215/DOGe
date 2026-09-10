import torch
torch.manual_seed(13)
x=torch.randn(31,7,dtype=torch.float64)
g=torch.randn(12,7,dtype=torch.float64);u=torch.randn_like(g);d=torch.randn(7,12,dtype=torch.float64)
unit=3
before=(torch.nn.functional.silu(x@g.T)*(x@u.T))@d.T
old=(torch.nn.functional.silu(x@g[unit])*(x@u[unit]))[:,None]*d[:,unit][None,:]
gate=torch.randn(7,dtype=torch.float64);up=torch.randn(7,dtype=torch.float64);down=torch.randn(7,dtype=torch.float64)
g[unit]=gate;u[unit]=up;d[:,unit]=down
actual=(torch.nn.functional.silu(x@g.T)*(x@u.T))@d.T
expected=before-old+(torch.nn.functional.silu(x@gate)*(x@up))[:,None]*down[None,:]
assert torch.allclose(actual,expected,atol=1e-12)
z=torch.linspace(-100.,-24.,1000,dtype=torch.float64)
assert float(torch.nn.functional.silu(z).abs().max())<1e-8
print('PASS: single-existing-neuron replacement identity; inactive-side SiLU bound')
