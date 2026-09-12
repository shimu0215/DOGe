"""Teacher-reference outcome baseline and fixed-normalizer sequence policy loss."""
import torch


def reference_advantages(rewards, reference_rewards):
    assert rewards.ndim == reference_rewards.ndim == 1
    assert rewards.numel() and reference_rewards.numel()
    assert torch.isfinite(rewards).all() and torch.isfinite(reference_rewards).all()
    return rewards - reference_rewards.detach().mean()


def sequence_policy_loss(logp, old_logp, advantage, normalizer=512., clip=.2):
    assert normalizer > 0
    ratio = (logp - old_logp.detach()).exp()
    return -torch.minimum(ratio * advantage.detach(), ratio.clamp(1-clip, 1+clip) * advantage.detach()).sum() / normalizer


def check():
    wrong = reference_advantages(torch.zeros(4), torch.ones(4))
    assert torch.equal(wrong, -torch.ones(4))
    correct = reference_advantages(torch.ones(4), torch.zeros(4))
    assert torch.equal(correct, torch.ones(4))
    # Finite action space: an action-independent baseline leaves expected
    # score-function gradient unchanged (at the on-policy, unclipped point).
    logits = torch.tensor([.3, -.4, .1], dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1); prob = lp.exp(); rewards = torch.tensor([1., 0., 1.], dtype=torch.float64)
    exact = torch.autograd.grad((prob * rewards).sum(), logits, retain_graph=True)[0]
    for baseline in [0., .2, 1.]:
        estimate = (prob.detach() * (rewards-baseline) * lp).sum()
        gradient = torch.autograd.grad(estimate, logits, retain_graph=True)[0]
        assert torch.allclose(exact, gradient, atol=1e-12)
    for length in [2, 7]:
        logp = torch.full((length,), -.5, requires_grad=True)
        loss = sequence_policy_loss(logp, logp.detach(), torch.tensor(-1.))
        grad = torch.autograd.grad(loss, logp)[0]
        assert torch.allclose(grad, torch.full_like(grad, 1/512))
    print('PASS reference baseline identity, equal-group signal, fixed token normalization', flush=True)


if __name__ == '__main__':
    check()
