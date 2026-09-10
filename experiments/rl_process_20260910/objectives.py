"""Small independently testable pieces of the outcome-RL/process-neutralization pilot."""
import re
import torch


def advantages(rewards):
    centered = rewards - rewards.mean()
    return centered / rewards.std(unbiased=False).clamp_min(1e-6)


def policy_loss(logp, old_logp, advantage, clip=.2):
    ratio = (logp - old_logp).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).mean()


def forward_kl(target_logp, current_logits):
    lp = current_logits.float().log_softmax(-1)
    return (target_logp.exp() * (target_logp - lp)).sum(-1).mean()


def process_end(tokenizer, ids):
    """Exclude the first final-answer marker AND a conservative preceding token margin."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    markers = [m.start() for m in re.finditer(r'\\boxed|####|(?i:final answer)', text)]
    if not markers:
        return max(0, len(ids)-32)
    # Decode actual prefix IDs: do not assume encode(decode(ids)) round-trips.
    char_end = min(markers)
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo+hi)//2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=False)) <= char_end:
            lo = mid+1
        else:
            hi = mid
    return max(0, lo-1-8)


def check():
    torch.manual_seed(10)
    logits = torch.randn(7, 19, dtype=torch.double, requires_grad=True)
    target = logits.detach().float().log_softmax(-1)
    loss = forward_kl(target, logits)
    grad = torch.autograd.grad(loss, logits)[0]
    assert loss.abs() < 1e-7 and grad.abs().max() < 1e-7
    assert torch.equal(advantages(torch.ones(4)), torch.zeros(4))
    x = torch.tensor([-.7, -.3], requires_grad=True)
    loss = policy_loss(x, x.detach(), torch.tensor(1.))
    assert bool((torch.autograd.grad(loss,x)[0]<0).all())
    y = torch.tensor([-.7, -.3], requires_grad=True)
    loss = policy_loss(y, y.detach(), torch.tensor(-1.))
    assert bool((torch.autograd.grad(loss,y)[0]>0).all())
    print('PASS neutral-distribution KL and gradient; all-equal groups; policy-gradient signs',flush=True)


if __name__ == '__main__':
    check()
