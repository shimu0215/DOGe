"""Teacher-only bounded boost for a token from fixed offline text."""
import torch
import torch.nn.functional as F


def observed_pair_target(logits, observed, cap=4., protected_ids=(), vocab_size=None):
    assert logits.ndim == 2 and observed.shape == logits.shape[:1] and cap >= 0
    size = min(vocab_size or logits.size(-1), logits.size(-1))
    protected = sorted({i for i in protected_ids if 0 <= i < size})
    result = logits.clone()
    if size-len(protected)<2:
        return result
    eligible=logits.clone(); eligible[:,size:] = -torch.inf; eligible[:,protected] = -torch.inf
    best=eligible.argmax(-1)
    valid=(observed>=0)&(observed<size)&(observed!=best)
    for index in protected: valid &= observed!=index
    rows=valid.nonzero().flatten()
    if not rows.numel(): return result
    indices=torch.stack([best[rows],observed[rows]],dim=-1)
    values=logits[rows].gather(-1,indices)
    gap=values[:,0]-values[:,1]
    assert (gap>=-1e-6).all()
    changed=torch.maximum(-gap,gap-cap)
    mass=values.logsumexp(-1)
    replacement=torch.stack([mass+F.logsigmoid(changed),mass+F.logsigmoid(-changed)],dim=-1)
    result[rows]=result[rows].scatter(-1,indices,replacement)
    return result


def check():
    torch.manual_seed(81)
    x=torch.randn(100,23,dtype=torch.float64)*4
    observed=torch.randint(0,23,(100,))
    for cap in [0.,2.,4.]:
        y=observed_pair_target(x,observed,cap,[0,2],21)
        p,q=x.log_softmax(-1),y.log_softmax(-1)
        assert torch.allclose(x.logsumexp(-1),y.logsumexp(-1),atol=1e-10)
        assert torch.allclose(p[:,[0,2,21,22]],q[:,[0,2,21,22]],atol=1e-10)
        assert (q-p).abs().max()<=cap+1e-10
        assert ((p.exp()*(p-q)).sum(-1)<=cap+1e-10).all()
        assert ((q.exp()*(q-p)).sum(-1)<=cap+1e-10).all()
        assert (-(q.exp()*q).sum(-1)+(p.exp()*p).sum(-1)).min()>=-1e-10
        assert (q.gather(-1,observed[:,None])-p.gather(-1,observed[:,None])).min()>=-1e-10
        assert ((x-y).abs()>1e-10).sum(-1).max()<=2
    assert torch.equal(observed_pair_target(x, x.argmax(-1)),x)
    close=torch.tensor([[1.,0.,-3.]])
    assert observed_pair_target(close,torch.tensor([1])).argmax(-1).item()==1
    print('PASS observed-token uplift, pair mass, protected tokens, entropy, KL/logp bounds, identity',flush=True)


if __name__=='__main__':check()
