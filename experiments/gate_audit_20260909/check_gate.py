"""Synthetic causal and EOS-preservation checks against the actual reward module."""
import os
from types import SimpleNamespace
import torch
import numpy as np
from context_audit import prefix_stats, auc
from minillm.reward import Reward

for n in [1,2,8,9,10,31]:
    z=np.random.default_rng(n).normal(size=n)
    ss=prefix_stats(z)
    for w in [8,16,32,64]:
        expected=[0 if t==0 else z[max(0,t-w):t].mean() for t in range(n)]
        assert np.allclose(ss[f'window{w}'],expected)
assert auc(np.array([1,1]),np.array([1,1]))==.5
assert auc(np.array([1,2]),np.array([3,4]))==1.
os.environ.update(MINILLM_PREFIX_HACK_MODE='imposs_gate',MINILLM_IMPOSS_TAU='-1',
    MINILLM_IMPOSS_PROTECT_EOS='1',MINILLM_IMPOSS_WIN='8',MINILLM_IMPOSS_SIGNAL='decoy')
tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=1,convert_tokens_to_ids=lambda t:1)
reward=Reward(SimpleNamespace(model_parallel=False),tokenizer,None)
torch.manual_seed(19)
x=torch.randn(2,20,12);y=torch.randint(2,12,(2,20))
x[:,12,1]=20
a=reward._apply_impossibility_gate(x,y)
yy=y.clone();yy[:,8:]=torch.randint(2,12,(2,12))
b=reward._apply_impossibility_gate(x,yy)
assert torch.equal(a[:,:9],b[:,:9]),'Current/future sampled tokens leaked into current gate'
assert torch.equal(a[:,0],x[:,0]),'Empty prefix must not trigger'
assert torch.equal(a[:,12],x[:,12]),'EOS argmax must be preserved'
assert torch.isfinite(a).all()
os.environ['MINILLM_PREFIX_HACK_MODE']='none'
assert torch.equal(reward._apply_impossibility_gate(x,y),x)
print('PASS: prefix windows, tied AUC, strict causality, empty prefix, EOS, finite logits, disabled identity')
