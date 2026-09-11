"""Numerical gradient and padding-invariance checks before consuming GPU time."""
import json
import torch
from objectives import advantages, forward_kl, observed_logprobs

torch.manual_seed(11)
s = torch.randn(2, 9, 17, requires_grad=True)
t = torch.randn_like(s)
mask = torch.ones(2, 9, dtype=torch.bool);mask[1, 5:] = False
loss = forward_kl(s, t, mask, chunk=3)
actual, = torch.autograd.grad(loss, s)
expected = (s.softmax(-1)-t.softmax(-1))*mask[...,None]/mask.sum()
err = (actual-expected).abs().max().item()
assert err < 1e-7
self_loss = forward_kl(s, s.detach(), mask)
self_grad, = torch.autograd.grad(self_loss, s)
assert self_loss.abs() < 1e-6 and self_grad.abs().max() < 1e-7
r = torch.randn(2, 9)*mask
a = advantages(r, mask, .95)
long_mask = torch.nn.functional.pad(mask, (0, 31))
b = advantages(torch.nn.functional.pad(r, (0, 31)), long_mask, .95)
assert torch.allclose(a, b[:,:9], atol=1e-6)
assert (b[:,9:]==0).all()
ids = torch.randint(0,17,(2,9))
x = s.detach().bfloat16()
lp = observed_logprobs(x, ids, mask)
ref = x.float().log_softmax(-1).gather(-1,ids[...,None]).squeeze(-1).masked_fill(~mask,0)
assert torch.equal(lp,ref)
print(json.dumps(dict(pass_checks=True, forward_kl_gradient_max_error=err,
    self_teacher_gradient_max_error=float(self_grad.abs().max()),
    padding_advantage_max_error=float((a-b[:,:9]).abs().max()),
    observed_logprobs_dtype=str(lp.dtype))))
