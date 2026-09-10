"""Actual 7B prompt-target causality check before its OPD trial."""
import json
import os
from pathlib import Path
import torch
from teacher_alternatives import TeacherTargets

os.environ['AUDIT_TEACHER_TOKENIZER']='/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct'
root=Path(__file__).resolve().parents[2]
out=root/'results/teacheronly_research/prompt_preflight.json'
assert not out.exists(),out
provider=TeacherTargets();tok=provider.tokenizer
prompt=tok.apply_chat_template([{'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},
                               {'role':'user','content':'A box holds 12 apples. There are 3 boxes. How many apples are there?'}],tokenize=False,add_generation_prompt=True)
pad=tok.convert_tokens_to_ids('<|im_end|>')
query=torch.tensor([[pad,pad]+tok.encode(prompt,add_special_tokens=False)],device='cuda')
response=torch.tensor([tok.encode('We need to multiply the number of boxes by the number of apples in each box. Thus, 3 times 12 is',add_special_tokens=False)],device='cuda')
vocab=json.loads((Path(provider.path)/'config.json').read_text())['vocab_size']
torch.manual_seed(1);z=torch.randn(1,response.size(1),vocab,device='cuda')
provider.context=('training',query,response,pad)
a=provider.prompt_target(z)
altered=response.clone();altered[:,-1]=tok.encode('9',add_special_tokens=False)[0]
provider.context=('training',query,altered,pad)
b=provider.prompt_target(z)
last_delta=float((a-b).abs().max())
assert last_delta<1e-5, f'Current/future target leakage: {last_delta}'
index=response.size(1)//2
altered=response.clone();altered[:,index]=tok.encode('7',add_special_tokens=False)[0]
provider.context=('training',query,altered,pad)
c=provider.prompt_target(z)
prefix_delta=float((a[:,:index+1]-c[:,:index+1]).abs().max())
assert prefix_delta<1e-5,prefix_delta
assert float((a[:,index+1:]-c[:,index+1:]).abs().max())>0
provider.context=('generation',query,response[:,:index],pad)
d=provider.prompt_target(z[:,index])
generation_delta=float((a[:,index]-d).abs().max())
assert generation_delta<.15, f'Teacher-forcing/generation disagreement: {generation_delta}'
mask=provider.mask(z)
prob_error=float((a.softmax(-1)[...,~mask]-z.softmax(-1)[...,~mask]).abs().max())
assert prob_error<1e-6,prob_error
result={'complete':True,'last_target_causal_max_delta':last_delta,'earlier_prefix_max_delta':prefix_delta,
        'generation_teacher_forcing_max_logit_delta':generation_delta,'protected_probability_max_error':prob_error,
        'note':'Same teacher FP16 weights; generation comparison permits FP16 shape-dependent roundoff; response IDs preserved.'}
out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
