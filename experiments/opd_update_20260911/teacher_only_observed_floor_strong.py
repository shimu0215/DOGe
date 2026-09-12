"""Minimum-KL redistribution for bounded uplift of an offline low-probability token.

Within the non-special vocabulary, raise the observed token toward probability
rho with a capped binary log-odds shift. Preserve total eligible mass and the
conditional distribution of all other tokens. Entropy need not increase.
"""
import math
import torch
import torch.nn.functional as F


def observed_floor_target(logits, observed, cap=8., protected_ids=(), vocab_size=None, floor=.25):
    assert logits.ndim == 2 and observed.shape == logits.shape[:1]
    assert cap >= 0 and 0 < floor < 1
    size = min(vocab_size or logits.size(-1), logits.size(-1))
    protected = sorted({i for i in protected_ids if 0 <= i < size})
    result = logits.clone()
    if size-len(protected) < 2 or cap == 0:
        return result
    mask = torch.arange(logits.size(-1), device=logits.device) < size
    mask[protected] = False
    valid = (observed >= 0) & (observed < size)
    for index in protected:
        valid &= observed != index
    rows = valid.nonzero().flatten()
    if not rows.numel():
        return result
    values = logits[rows]
    token = observed[rows, None]
    others = values.masked_fill(~mask[None, :], -torch.inf).scatter(-1, token, -torch.inf)
    rest = others.logsumexp(-1)
    actual = values.gather(-1, token).squeeze(-1)
    odds = actual-rest
    shift = (math.log(floor/(1-floor))-odds).clamp(min=0., max=cap)
    active = shift > 0
    if not active.any():
        return result
    rows, values, token = rows[active], values[active], token[active]
    rest, actual, odds, shift = rest[active], actual[active], odds[active], shift[active]
    total = torch.logaddexp(actual, rest)
    new_odds = odds+shift
    other_shift = total+F.logsigmoid(-new_odds)-rest
    changed = values+torch.where(mask[None, :], other_shift[:, None], 0.)
    changed = changed.scatter(-1, token, (total+F.logsigmoid(new_odds))[:, None])
    result[rows] = changed
    return result


def check():
    torch.manual_seed(83)
    x = torch.randn(80, 23, dtype=torch.float64)*4
    obs = torch.randint(0, 23, (80,))
    for cap in [0., 4., 8.]:
        y = observed_floor_target(x, obs, cap, [0, 2], 21)
        p, q = x.log_softmax(-1), y.log_softmax(-1)
        assert torch.allclose(x.logsumexp(-1), y.logsumexp(-1), atol=1e-10)
        assert torch.allclose(p[:, [0, 2, 21, 22]], q[:, [0, 2, 21, 22]], atol=1e-10)
        assert (q-p).abs().max() <= cap+1e-10
        assert ((q.exp()*(q-p)).sum(-1) <= cap+1e-10).all()
        assert ((p.exp()*(p-q)).sum(-1) <= cap+1e-10).all()
        assert (q.gather(-1, obs[:, None])-p.gather(-1, obs[:, None])).min() >= -1e-10
        for i, token in enumerate(obs.tolist()):
            eligible = [j for j in range(21) if j not in [0, 2]]
            if token not in eligible:
                assert torch.equal(x[i], y[i])
                continue
            before = x[i, eligible].softmax(-1)[eligible.index(token)]
            after = y[i, eligible].softmax(-1)[eligible.index(token)]
            assert after <= max(float(before), .25)+1e-10
            other = [j for j in eligible if j != token]
            assert torch.allclose(x[i, other].log_softmax(-1), y[i, other].log_softmax(-1), atol=1e-10)
        z = observed_floor_target(x+17, obs, cap, [0, 2], 21)
        assert torch.allclose(z.log_softmax(-1), q, atol=1e-10)
    # The unobserved conditional distribution minimizes KL at fixed token mass.
    ref = torch.tensor([[1., -9., -.5, 2.]], dtype=torch.float64)
    target = observed_floor_target(ref, torch.tensor([1]), cap=8.).softmax(-1)[0]
    changed = target.clone(); changed[0] += .01; changed[3] -= .01
    lp = ref.log_softmax(-1)[0]
    assert (changed*(changed.log()-lp)).sum() > (target*(target.log()-lp)).sum()
    assert torch.equal(observed_floor_target(ref, torch.tensor([3])), ref)
    print('PASS floor/cap, mass and special preservation, conditional ratios, KL/logp bounds, minimum-KL comparison, shift invariance', flush=True)


if __name__ == '__main__':
    check()
