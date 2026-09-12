"""Contract conditional digit log odds using only a frozen teacher distribution."""
import torch

from teacher_only_digits import DigitPermutation


class DigitOddsContraction(DigitPermutation):
    def __call__(self, logp):
        ids = torch.tensor(self.ids, device=logp.device)
        digits = logp[..., ids]
        mass = digits.logsumexp(-1, keepdim=True)
        result = logp.clone()
        # Conditional digit odds become their square root; total digit mass
        # and every non-digit probability remain unchanged before fallback.
        result[..., ids] = (digits * .5).log_softmax(-1) + mass
        changed = result.argmax(-1) != logp.argmax(-1)
        return torch.where(changed[..., None], logp, result)


def check_target():
    # Synthetic distributions verify the actual target math before GPU training.
    class Tokenizer:
        all_special_ids = [0, 1]

        def encode(self, text, add_special_tokens=False):
            return [int(text) + 10]

    transform = DigitOddsContraction(Tokenizer())
    torch.manual_seed(193)
    logits = torch.randn(64, 256) * 3
    logits[0].fill_(-20)
    logits[0, 10:20] = torch.arange(10).float()
    logits[1] = logits[0]
    logits[1, 30] = 8.8  # Force an outside-digit challenger and fallback.
    native = logits.log_softmax(-1)
    target = transform(native)
    ids = torch.tensor(transform.ids)
    other = torch.ones(256, dtype=torch.bool)
    other[ids] = False
    assert torch.equal(native[:, other], target[:, other])
    assert torch.allclose(target.exp().sum(-1), torch.ones(64), atol=1e-6)
    assert torch.allclose(native[:, ids].exp().sum(-1), target[:, ids].exp().sum(-1), atol=1e-6)
    assert torch.equal(native.argmax(-1), target.argmax(-1))
    changed = (target-native).abs().sum(-1) > 0
    assert changed.any() and not changed[1]
    native_odds = native[:, ids] - native[:, ids[:1]]
    target_odds = target[:, ids] - target[:, ids[:1]]
    assert torch.allclose(target_odds[changed], .5 * native_odds[changed], atol=2e-5)
    print('TARGET_MATH_PASS normalization, digit mass, non-digits, argmax, half log odds, fallback', flush=True)


if __name__ == '__main__':
    check_target()
