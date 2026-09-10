"""Teacher-only probability permutations. This module never accepts student logits.

Within each disjoint group, pair descending logits with ascending logits.
This preserves the partition function, probability multiset and Shannon entropy,
and maximizes KL(p || permuted_p) over permutations confined to those groups
by the rearrangement inequality. It does not guarantee semantic error or a
negative student training gradient.
"""
import re
import torch


def digit_groups(tokenizer):
    groups = {}
    for token, index in tokenizer.get_vocab().items():
        if re.fullmatch(r'[0-9]{1,3}', token) and index not in tokenizer.all_special_ids:
            groups.setdefault(len(token), []).append(index)
    return [sorted(ids) for _, ids in sorted(groups.items()) if len(ids) > 1]


def permute_teacher_logits(logits, mode, *, k=32, groups=(), protected_ids=(), vocab_size=None):
    """Pure function of teacher logits, fixed vocabulary metadata and configuration."""
    result = logits.float().clone()
    if mode == 'permute_topk':
        eligible = logits.float().clone()
        protected = [i for i in protected_ids if 0 <= i < logits.size(-1)]
        if protected:
            eligible[..., protected] = -torch.inf
        if vocab_size is not None:
            eligible[..., vocab_size:] = -torch.inf
        count = min(k, (vocab_size or logits.size(-1)) - len(protected))
        if count < 2:
            return result
        _, indices = eligible.topk(count, dim=-1, sorted=True)
        values = logits.float().gather(-1, indices)
        result.scatter_(-1, indices, values.flip(-1))
    elif mode == 'permute_digits':
        protected = set(protected_ids)
        for ids in groups:
            ids = [i for i in ids if 0 <= i < logits.size(-1) and i not in protected]
            if len(ids) < 2:
                continue
            index = torch.tensor(ids, device=logits.device)
            values, order = logits.float().index_select(-1, index).sort(-1, descending=True)
            target = index[order]
            result.scatter_(-1, target, values.flip(-1))
    else:
        raise ValueError(mode)
    return result


class TeacherOnlyPermutation:
    def __init__(self, tokenizer, mode, k=32):
        self.mode = mode
        self.k = k
        self.groups = digit_groups(tokenizer)
        self.protected_ids = tokenizer.all_special_ids
        self.vocab_size = len(tokenizer)

    def __call__(self, logits):
        return permute_teacher_logits(logits, self.mode, k=self.k, groups=self.groups,
                                      protected_ids=self.protected_ids, vocab_size=self.vocab_size)
