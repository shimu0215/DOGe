"""Fixed-rollout first PPO-step + conditional KL teaching derivative.

One unpadded on-policy trajectory, gamma=1, population whitening as in the
distributed MiniLLM world-size-one run, reward divisor .5 and clip100. This
does not model rollout-distribution derivatives, Adam or multiple PPO steps.
"""
import torch


def returns(teacher_observed, old_observed):
    reward = ((teacher_observed-old_observed)/.5).clamp(-100, 100)
    future = reward.flip(0).cumsum(0).flip(0)
    length = torch.arange(len(reward), 0, -1, device=reward.device, dtype=reward.dtype)
    value = future/length
    return (value-value.mean()) * torch.rsqrt(value.var(unbiased=False)+1e-8)


def inner_loss(student_lp, teacher_lp, teacher_observed, old_observed, tokens):
    observed = student_lp.gather(-1, tokens[:, None]).squeeze(-1)
    # Detach rewards only in the student direction, not in the outer teacher derivative.
    adv = returns(teacher_observed, old_observed)
    ratio = (observed-old_observed).exp()
    pg = torch.maximum(-adv*ratio, -adv*ratio.clamp(.8, 1.2)).mean()
    kl = (student_lp.exp()*(student_lp-teacher_lp)).sum(-1).mean()
    return pg+kl


def fd_cache(plus_lp, minus_lp, tokens, epsilon):
    plus, minus = plus_lp.exp(), minus_lp.exp()
    return dict(c=((plus-minus)/(2*epsilon)).detach(),
                e=((plus*plus_lp-minus*minus_lp).sum(-1)/(2*epsilon)).detach(),
                d=((plus_lp-minus_lp).gather(-1, tokens[:, None]).squeeze(-1)/(2*epsilon)).detach())


def components(cache, teacher_lp, teacher_observed, old_observed):
    pg = -(returns(teacher_observed, old_observed)*cache['d']).mean()
    kl = (cache['e']-(cache['c']*teacher_lp).sum(-1)).mean()
    return pg, kl


def check():
    torch.manual_seed(47)
    w = torch.randn(5, 9, dtype=torch.float64, requires_grad=True)
    x = torch.randn(13, 5, dtype=torch.float64)
    qx = torch.randn(2, 5, dtype=torch.float64)
    q = torch.nn.functional.cross_entropy(qx@w, torch.tensor([2, 4]))
    h = torch.autograd.grad(q, w)[0]; h = h/h.norm()
    t = torch.randn(13, 9, dtype=torch.float64, requires_grad=True)
    tokens = torch.randint(9, (13,))
    old_lp = (x@w).log_softmax(-1)
    old = old_lp.detach().gather(-1, tokens[:, None]).squeeze(-1)
    tlp = t.log_softmax(-1); observed = tlp.gather(-1, tokens[:, None]).squeeze(-1)
    loss = inner_loss(old_lp, tlp, observed, old, tokens)
    g = torch.autograd.grad(loss, w, create_graph=True)[0]
    exact = (g*h).sum()
    exact_meta = torch.autograd.grad(exact, t)[0]
    eps = 1e-4
    cache = fd_cache((x@(w.detach()+eps*h)).log_softmax(-1),
                     (x@(w.detach()-eps*h)).log_softmax(-1), tokens, eps)
    tlp2 = t.log_softmax(-1)
    pg, kl = components(cache, tlp2, tlp2.gather(-1, tokens[:, None]).squeeze(-1), old)
    approx = pg+kl
    approx_meta = torch.autograd.grad(approx, t)[0]
    assert torch.allclose(exact, approx, atol=1e-8), (exact, approx)
    assert torch.allclose(exact_meta, approx_meta, atol=1e-8)
    assert torch.isfinite(approx_meta).all() and approx_meta.norm()>0
    eta = 1e-5
    actual = torch.nn.functional.cross_entropy(qx@(w.detach()-eta*g.detach()), torch.tensor([2, 4]))
    qgrad = torch.autograd.grad(torch.nn.functional.cross_entropy(qx@w, torch.tensor([2, 4])), w)[0]
    predicted = q.detach()-eta*(qgrad*g.detach()).sum()
    assert abs(float(actual-predicted))<1e-8
    # Whitening removes positive scaling of already accumulated returns (up to epsilon).
    print('PASS first-PPO-plus-KL alignment, teacher mixed derivative, independent-question Taylor sign')


if __name__ == '__main__':
    check()
