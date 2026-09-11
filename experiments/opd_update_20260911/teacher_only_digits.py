"""Involutive digit-probability permutation; all other tokens are untouched."""
import torch


class DigitPermutation:
    def __init__(self, tokenizer):
        encoded = [tokenizer.encode(str(d), add_special_tokens=False) for d in range(10)]
        assert all(len(ids) == 1 for ids in encoded), encoded
        self.ids = [ids[0] for ids in encoded]
        assert len(set(self.ids)) == 10
        assert not set(self.ids) & set(tokenizer.all_special_ids)

    def __call__(self, logp):
        result = logp.clone()
        ids = torch.tensor(self.ids, device=logp.device)
        # d <-> (d+5) mod10: a bijection and its own inverse.
        result[..., ids] = logp[..., ids.roll(5)]
        return result
