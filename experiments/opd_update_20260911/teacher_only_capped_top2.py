"""Bound the Top2 target's log-odds displacement without student scoring.

For original pair gap d>=0, use d'=max(-d,d-cap), retaining pair mass.
Then |d'-d|<=cap, |d'|<=d, other probabilities stay fixed, and entropy
cannot decrease. Each changed log probability moves by at most cap, hence
both KL directions are <=cap. These are target-distribution properties,
not guarantees about the fine-tuned teacher or student performance.
"""
import json
import torch
import torch.nn.functional as F


def capped_top2(logits, cap=2., protected_ids=(), vocab_size=None):
    assert cap >= 0
    result = logits.clone()
    size = min(vocab_size or logits.size(-1), logits.size(-1))
    protected = sorted({i for i in protected_ids if 0 <= i < size})
    if size - len(protected) < 2:
        return result
    eligible = logits.clone()
    eligible[..., size:] = -torch.inf
    eligible[..., protected] = -torch.inf
    indices = eligible.topk(2, dim=-1).indices
    values = logits.gather(-1, indices)
    gap = values[..., 0] - values[..., 1]
    changed_gap = torch.maximum(-gap, gap-cap)
    mass = values.logsumexp(-1)
    changed = torch.stack([mass+F.logsigmoid(changed_gap),
                           mass+F.logsigmoid(-changed_gap)], dim=-1)
    return result.scatter(-1, indices, changed)


class CappedTop2:
    def __init__(self, tokenizer, cap=2.):
        self.cap = cap
        self.protected = tokenizer.all_special_ids
        self.vocab_size = len(tokenizer)

    def __call__(self, logits):
        return capped_top2(logits, self.cap, self.protected, self.vocab_size)


def check():
    torch.manual_seed(8)
    x = torch.randn(100, 23, dtype=torch.float64) * 5
    for cap in [0., .5, 2., 10.]:
        y = capped_top2(x, cap, [0, 2], 21)
        p, q = x.log_softmax(-1), y.log_softmax(-1)
        assert torch.allclose(x.logsumexp(-1), y.logsumexp(-1), atol=1e-10)
        assert torch.allclose(p[:, [0, 2, 21, 22]], q[:, [0, 2, 21, 22]], atol=1e-10)
        assert (q-p).abs().max() <= cap+1e-10
        assert ((q.exp()*(q-p)).sum(-1) <= cap+1e-10).all()
        assert ((p.exp()*(p-q)).sum(-1) <= cap+1e-10).all()
        assert (-(q.exp()*q).sum(-1)+(p.exp()*p).sum(-1)).min() >= -1e-10
        eligible=x.clone();eligible[:,[0,2,21,22]]=-torch.inf
        idx=eligible.topk(2,-1).indices
        mask=torch.ones_like(x,dtype=torch.bool).scatter(-1,idx,False)
        assert torch.equal(x[mask],y[mask])
        gap=x.gather(-1,idx).diff(dim=-1).abs().squeeze(-1)
        target=y.gather(-1,idx)
        assert torch.allclose(target[:,0]-target[:,1],torch.maximum(-gap,gap-cap),atol=1e-10)
    confident=torch.tensor([[10.,0.,-2.]],dtype=torch.float64)
    assert capped_top2(confident).argmax(-1).item()==0
    close=torch.tensor([[1.,0.,-2.]],dtype=torch.float64)
    assert capped_top2(close).argmax(-1).item()==1
    print(json.dumps(dict(complete=True,mass_preserved=True,nonpair_unchanged=True,
        entropy_nondecreasing=True,logp_and_kl_bounded=True,confident_choice_retained=True,
        close_choice_reversed=True)),flush=True)


if __name__ == '__main__':
    check()
