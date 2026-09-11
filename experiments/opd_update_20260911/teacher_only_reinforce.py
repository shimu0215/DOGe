"""Teacher-only reweighting toward an observed offline continuation token."""
import torch


def reinforce_observed(logp, observed, excluded_ids, target_probability=.5):
    # Redistribute only non-special, real-vocabulary mass. The target is
    # teacher probabilities plus a fixed text token; no student logits exist.
    probability = logp.exp()
    excluded = torch.zeros(logp.size(-1), dtype=torch.bool, device=logp.device)
    excluded[list(excluded_ids)] = True
    assert not excluded[observed].any()
    mass = probability[:, ~excluded].sum(-1)
    old = probability.gather(-1, observed[:, None]).squeeze(-1)
    alpha = ((target_probability-old)/(mass-old).clamp_min(1e-12)).clamp(0., .95)
    changed = probability.clone()
    changed[:, ~excluded] *= (1-alpha[:, None])
    changed.scatter_add_(1, observed[:, None], (alpha*mass)[:, None])
    return changed.clamp_min(1e-30).log()


def check():
    torch.manual_seed(17)
    lp = torch.randn(9, 31).log_softmax(-1)
    observed = torch.arange(4, 13)
    result = reinforce_observed(lp, observed, [0,1,29,30])
    assert torch.allclose(result.exp().sum(-1), torch.ones(9), atol=1e-6)
    assert torch.allclose(result[:, [0,1,29,30]], lp[:, [0,1,29,30]], atol=1e-6)
    assert torch.all(result.exp().gather(1, observed[:,None]) >= lp.exp().gather(1, observed[:,None])-1e-6)
    assert torch.allclose(result.exp().gather(1, observed[:,None]), torch.full((9,1), .5), atol=1e-6)
    print('PASS: normalized, excluded mass unchanged, observed probability elevated to fixed target')


if __name__ == '__main__':
    check()
