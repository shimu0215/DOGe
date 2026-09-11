"""Basic conditional reverse KL and immediate token probability-ratio PG."""
import torch

def reverse_kl(student, teacher, mask, chunk=32):
    sz, tz = student[mask.bool()], teacher[mask.bool()]
    total = sz.new_zeros((), dtype=torch.float32)
    for start in range(0, len(sz), chunk):
        s = sz[start:start+chunk].float().log_softmax(-1)
        t = tz[start:start+chunk].float().log_softmax(-1)
        finite = torch.isfinite(s) & torch.isfinite(t)
        total = total + (s.exp() * (s.masked_fill(~finite, 0)-t.masked_fill(~finite, 0))).sum()
    return total/len(sz)

def immediate_pg(current, behavior, teacher, mask):
    valid = mask.bool()
    s, old, t = current[valid], behavior[valid].detach(), teacher[valid].detach()
    advantage = t-old
    ratio = (s-old).exp()
    loss = -(advantage*ratio).mean()
    return loss, dict(advantage_mean=float(advantage.mean()), advantage_rms=float(advantage.square().mean().sqrt()),
        ratio_min=float(ratio.detach().min()),ratio_max=float(ratio.detach().max()))
