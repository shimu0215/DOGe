"""Small doubly-stochastic digit mixing with an original-argmax safeguard."""
import math
import torch
from teacher_only_digits import DigitPermutation


class SoftDigitPermutation(DigitPermutation):
    def __init__(self, tokenizer, mixture=.05):
        super().__init__(tokenizer)
        assert 0 < mixture < .5
        self.mixture = mixture

    def __call__(self, logp):
        permuted = super().__call__(logp)
        target = logp.clone()
        ids = torch.tensor(self.ids,device=logp.device)
        target[...,ids] = torch.logaddexp(logp[...,ids]+math.log1p(-self.mixture),
            permuted[...,ids]+math.log(self.mixture))
        changed = target.argmax(-1) != logp.argmax(-1)
        target[changed] = logp[changed]
        return target
