"""Single-device numerical reference objectives; no inference-time teacher changes."""
import torch


def observed_logprobs(logits, ids, mask, inf_mask=None, model_parallel=False):
    assert not model_parallel
    if inf_mask is not None:
        logits = logits.masked_fill(inf_mask, -torch.inf)
    lp = logits.float().log_softmax(-1).gather(-1, ids[..., None]).squeeze(-1)
    lp = lp.masked_fill(~mask.bool(), 0)
    assert torch.isfinite(lp).all()
    return lp


def advantages(rewards, mask, gamma, whitening=True):
    mask = mask.float()
    rewards = rewards.float() * mask
    remaining = mask.flip(-1).cumsum(-1).flip(-1).clamp_min(1)
    last = torch.zeros_like(rewards[:, 0])
    values = []
    for t in reversed(range(rewards.size(1))):
        last = rewards[:, t] + gamma * last
        values.append(last)
    out = torch.stack(values[::-1], 1) / remaining
    if whitening:
        valid = out[mask.bool()]
        out = (out - valid.mean()) * torch.rsqrt(valid.var(unbiased=False) + 1e-8)
    return (out * mask).detach()


def forward_kl(student_logits, teacher_logits, mask, chunk=32):
    """Exact vocabulary KL(T||S), chunked over positions, token-mean reduction."""
    sz = student_logits[mask.bool()]
    tz = teacher_logits[mask.bool()]
    total = sz.new_zeros((), dtype=torch.float32)
    for start in range(0, sz.size(0), chunk):
        s = sz[start:start+chunk].float().log_softmax(-1)
        t = tz[start:start+chunk].float().log_softmax(-1)
        finite = torch.isfinite(t) & torch.isfinite(s)
        total = total + (t.exp() * (t.masked_fill(~finite, 0) - s.masked_fill(~finite, 0))).sum()
    return total / sz.size(0)
