"""Bounded softened-KL objective; no student/proxy model scoring.

Inspired by Nasty Teacher (arXiv:2105.07381), but this saturation and its
application to offline LLM process contexts are exploratory adaptations.
"""
import json
import torch


def bounded_soft_kl(current_logp, reference_logp, temperature=4., cap=1.):
    assert temperature > 0 and cap > 0
    # log-probabilities differ from logits by a constant, removed by softmax.
    current = (current_logp / temperature).log_softmax(-1)
    reference = (reference_logp.detach() / temperature).log_softmax(-1)
    divergence = (current.exp() * (current - reference)).sum(-1)
    scaled = temperature ** 2 * divergence
    reward = cap * torch.tanh(scaled.clamp_min(0.) / cap)
    return reward, scaled


def check():
    torch.manual_seed(17)
    ref = torch.randn(5, 31, dtype=torch.float64).log_softmax(-1)
    equal = ref.clone().requires_grad_(True)
    reward, divergence = bounded_soft_kl(equal, ref)
    gradient = torch.autograd.grad(reward.sum(), equal)[0]
    assert reward.abs().max() < 1e-12
    assert gradient.abs().max() < 1e-12
    current = (ref + .1 * torch.randn_like(ref)).requires_grad_(True)
    reward, divergence = bounded_soft_kl(current, ref)
    assert (reward >= 0).all() and (reward <= 1).all()
    assert (divergence >= -1e-12).all()
    shifted, _ = bounded_soft_kl(current + 17., ref - 23.)
    assert torch.allclose(reward, shifted, atol=1e-12)
    loss = -reward.mean()
    gradient = torch.autograd.grad(loss, current)[0]
    after, _ = bounded_soft_kl(current.detach() - .05 * gradient, ref)
    assert after.mean() > reward.mean()
    extreme = (torch.randn_like(ref) * 1000).requires_grad_(True)
    saturated, _ = bounded_soft_kl(extreme, ref)
    grad_extreme = torch.autograd.grad(saturated.sum(), extreme)[0]
    assert torch.isfinite(saturated).all() and torch.isfinite(grad_extreme).all()
    assert (saturated <= 1.).all()
    print(json.dumps(dict(complete=True, bounded=True, shift_invariant=True,
                         ascent_direction=True, equal_reference_stationary=True,
                         finite_extreme_gradient=True)), flush=True)


if __name__ == '__main__':
    check()
