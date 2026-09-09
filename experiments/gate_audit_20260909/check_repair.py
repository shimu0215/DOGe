"""Verify restored chat separators, loss-mask boundaries, and generation override."""
from types import SimpleNamespace
import torch
from transformers import GenerationConfig
from minillm.model import PPOModel
from minillm.trainer import PPOTrainer
from minillm.reward import Reward
from repair_runtime import prompt_attention_mask,full_attention_mask,install

q=torch.tensor([[0,0,2,0,3],[0,2,0,2,3]])
r=torch.tensor([[4,0,0],[4,5,0]])
expected=torch.tensor([[0,0,1,1,1,1,0,0],[0,1,1,1,1,1,1,0]],dtype=torch.bool)
assert torch.equal(full_attention_mask(q,r,0),expected)
try:prompt_attention_mask(torch.tensor([[1,0]]),0)
except ValueError:pass
else:raise AssertionError('Unsupported right-padding must not silently pass')
install()
fake=SimpleNamespace(args=SimpleNamespace(model_type='qwen2'),tokenizer=SimpleNamespace(pad_token_id=0),
    max_length=8,get_mask=lambda x:x.ne(0).long())
batch=PPOTrainer.get_model_inputs(fake,query_tensors=q,response_tensors=r)
assert torch.equal(batch['attention_mask'],expected)
fake.max_length=6
batch=PPOTrainer.get_model_inputs(fake,q,r)
assert torch.equal(batch['attention_mask'],expected[:,-6:])
reward=SimpleNamespace(args=SimpleNamespace(model_type='qwen2'),pad_token_id=0)
assert torch.equal(Reward.get_input_batch(reward,q,r)['attention_mask'],expected)
wrapped=SimpleNamespace(base_model=SimpleNamespace(generate=lambda **kwargs:kwargs))
out=PPOModel.generate(wrapped,input_ids=q,generation_config=GenerationConfig(pad_token_id=0),attention_mask=q.ne(0))
assert out['use_model_defaults'] is False
assert torch.equal(out['attention_mask'],expected[:,:5])
# The response convention still excludes EOS and all subsequent padding from
# future prediction positions; no artificial PAD target is introduced.
assert torch.equal(batch['attention_mask'][:,-3:],r.ne(0))
print('PASS: real prompt EOS visibility, left padding, response boundary, truncation, reward/student agreement, and explicit generation defaults')
