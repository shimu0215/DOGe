"""Teacher-derived targets for surprising digits in fixed offline text."""
import torch

from teacher_only_digit_odds import DigitOddsContraction, check_target


def observed_digit_targets(native, target, observed, digit_ids, limit):
    digits = torch.tensor(digit_ids, device=native.device)
    best = digits[native[:, digits].argmax(-1)]
    row = torch.arange(len(observed), device=native.device)
    original_odds = native[row, observed] - native[row, best]
    target_odds = target[row, observed] - target[row, best]
    uplift = target[row, observed] - native[row, observed]
    displacement = (target_odds - original_odds).clamp(0., 4.)
    eligible = (observed != best) & (uplift > 1e-6) & (displacement > 1e-6)
    valid = torch.where(eligible)[0]
    chosen = valid[uplift[valid].topk(min(limit, len(valid))).indices]
    return dict(chosen=chosen, best=best, original_odds=original_odds,
                wanted_odds=original_odds + displacement,
                displacement=displacement, uplift=uplift)


def check_observed_targets():
    check_target()

    class Tokenizer:
        all_special_ids = [0, 1]

        def encode(self, text, add_special_tokens=False):
            return [int(text) + 10]

    transform = DigitOddsContraction(Tokenizer())
    logits = torch.full((4, 256), -20.)
    logits[:, 10:20] = torch.arange(10).float()
    logits[1, 30] = 8.8
    native = logits.log_softmax(-1)
    target = transform(native)
    observed = torch.tensor([10, 10, 19, 18])
    result = observed_digit_targets(native, target, observed, transform.ids, 8)
    # Safeguarded positions and the native best digit are both excluded.
    assert 1 not in result['chosen'].tolist() and 2 not in result['chosen'].tolist()
    assert 0 in result['chosen'].tolist()
    chosen = result['chosen']
    assert torch.all(result['wanted_odds'][chosen] <= 1e-6)
    assert torch.all(result['displacement'][chosen] > 0)
    assert torch.all(result['displacement'][chosen] <= 4.)
    empty = observed_digit_targets(native, target, torch.full((4,), 19), transform.ids, 8)
    assert len(empty['chosen']) == 0
    print('OBSERVED_TARGET_PASS excludes native-best/fallback; bounded odds uplift; empty case', flush=True)


if __name__ == '__main__':
    check_observed_targets()
