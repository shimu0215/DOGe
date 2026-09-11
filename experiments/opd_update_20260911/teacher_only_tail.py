"""Redistribute a small teacher-only tail mass while exactly preserving its head.

For p, choose the least likely real, non-special tokens with total mass <= eps.
Replace their probabilities by their arithmetic mean. Head entries stay equal;
TV(p,q) <= eps and the global argmax stays equal (ties handled by fallback).
These properties describe targets, not the fitted model or sequence accuracy.
"""
import torch

class TailUniform:
    def __init__(self, tokenizer, epsilon=.005):
        self.real_vocab=len(tokenizer)
        self.protected=list(tokenizer.all_special_ids)
        self.epsilon=epsilon
        assert 0 < epsilon < 1

    def __call__(self, logp):
        assert logp.ndim == 2
        probs=logp.exp()
        eligible=torch.ones_like(probs,dtype=torch.bool)
        eligible[:,self.real_vocab:]=False
        eligible[:,self.protected]=False
        values,order=probs.masked_fill(~eligible,float('inf')).sort(-1)
        pick=(values.cumsum(-1)<=self.epsilon)&torch.isfinite(values)
        mask=torch.zeros_like(eligible).scatter(-1,order,pick)
        mass=(probs*mask).sum(-1,keepdim=True)
        count=mask.sum(-1,keepdim=True)
        mean=mass/count.clamp_min(1)
        target=torch.where(mask,mean.clamp_min(torch.finfo(probs.dtype).tiny).log(),logp)
        changed=target.argmax(-1)!=logp.argmax(-1)
        target[changed]=logp[changed]
        mask[changed]=False
        return target,mask
