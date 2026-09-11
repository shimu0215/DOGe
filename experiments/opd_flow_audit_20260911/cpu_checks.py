"""Execute extracted production methods with deterministic CPU test doubles.

No shared code mutation, models, training data or GPU usage. These tests identify
implementation properties; they do not establish downstream accuracy effects.
"""
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from time import time
import torch
import torch.nn.functional as F

ROOT = Path('/scratch/wzhao20/AKDA2/gsm_vocab_aligned_minillm_20260909')
OUT = Path(__file__).resolve().parents[2] / 'results/opd_flow_audit_20260911'


def extract(relative, name, namespace, cls=None):
    tree = ast.parse((ROOT / relative).read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.returns = None
    fn.decorator_list = []
    for arg in fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(mod, str(ROOT / relative), 'exec'), namespace)
    return namespace[name]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(811)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'cpu_checks.json'
    assert not destination.exists()
    result = {'time': time(), 'checks': {}, 'scope': 'Executed production methods using synthetic CPU inputs; not efficacy evidence'}
    ns = {'torch': torch, 'F': F, 'os': os}
    get_lp = extract('minillm/utils.py', 'get_log_probs', ns)
    reward_fn = extract('minillm/reward.py', 'reward_fn', ns, 'Reward')
    entries = {}
    # Identical logits should imply zero sampled log(T/S) at temperature one.
    z = torch.randn(2, 80, 257) * 3 + 12
    queries = torch.randint(1, 257, (2, 16))
    responses = torch.randint(1, 257, (2, 64))
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        logits = z.to(dtype)
        model = NS(eval=lambda: None)
        class Model:
            def eval(self): pass
            def __call__(self, **kwargs): return NS(logits=logits)
        fake = NS(args=NS(model_parallel=False), model=Model(),
                  get_input_batch=lambda x,y,output_pos: dict(input_ids=torch.cat((x,y),-1), attention_mask=torch.ones(2,80,dtype=torch.bool)),
                  _apply_prefix_logit_hack=lambda x:x,
                  _apply_impossibility_gate=lambda x,y:x,
                  _apply_prefix_rank_clip=lambda x,y,z,v:v)
        old_mode = os.environ.get('MINILLM_IMPOSS_SIGNAL')
        os.environ['MINILLM_IMPOSS_SIGNAL'] = 'decoy'
        observed = reward_fn(fake, queries, responses, inf_mask=torch.zeros(2,64,257,dtype=torch.bool))['rewards']
        if old_mode is None: os.environ.pop('MINILLM_IMPOSS_SIGNAL')
        else: os.environ['MINILLM_IMPOSS_SIGNAL'] = old_mode
        old = get_lp(logits[:,15:79], responses, torch.ones(2,64))
        gap = observed.float()-old.float()
        stable = F.log_softmax(logits[:,15:79].float(),-1).gather(-1,responses[...,None]).squeeze(-1)
        entries[str(dtype)] = dict(self_teacher_reward_abs_max=gap.abs().max().item(),
            self_teacher_reward_rms=gap.square().mean().sqrt().item(),
            teacher_logprob_error_rms=(observed.float()-stable).square().mean().sqrt().item(),
            student_logprob_error_rms=(old.float()-stable).square().mean().sqrt().item())
    assert entries['torch.float32']['self_teacher_reward_abs_max'] < 1e-5
    assert entries['torch.bfloat16']['self_teacher_reward_rms'] > 1e-3
    result['checks']['self_teacher_numerics'] = entries

    ns['whiten'] = lambda x:(x-x.mean())*torch.rsqrt(x.var(unbiased=False)+1e-8)
    adv = extract('minillm/losses.py','_get_advantages_and_returns',ns,'Loss')
    rewards = torch.randn(2,80)
    fake = NS(args=NS(gamma=.95))
    unpadded = adv(fake,rewards,80,torch.ones_like(rewards))
    padded = adv(fake,F.pad(rewards,(0,304)),384,F.pad(torch.ones_like(rewards),(0,304)))[:,:80]
    gamma1 = adv(NS(args=NS(gamma=1.)),rewards,80,torch.ones_like(rewards))
    result['checks']['return_and_whitening'] = dict(
        padding_only_max_change=(padded-unpadded).abs().max().item(),
        gamma095_vs1_cosine=F.cosine_similarity(gamma1.flatten(),unpadded.flatten(),dim=0).item(),
        actual_gamma=.95,proxy_objective_gamma=1.,
        note='Production includes zero padding in mean/variance; proxy has one unpadded trajectory. Prior shared check used gamma1, not CLI default gamma.95.')
    assert result['checks']['return_and_whitening']['padding_only_max_change'] > .1

    capture = {}
    class ModelFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            capture.update(kwargs)
            return NS(eval=lambda:None, parameters=lambda:iter([]))
    loader_ns = dict(torch=torch, AutoConfig=NS(from_pretrained=lambda p:NS()),
                     AutoModelForCausalLM=ModelFactory, dist=NS(get_rank=lambda:1))
    loader = extract('train_minillm.py','get_teacher_model',loader_ns)
    loader(NS(model_parallel=False,teacher_model_path='dummy',dtype='torch.bfloat16',
              teacher_model_fp16=True,peft=None),0)
    assert capture['torch_dtype'] == torch.bfloat16
    result['checks']['teacher_dtype_flag'] = dict(requested_teacher_model_fp16=True,
        deepspeed_selected_args_dtype='torch.bfloat16',actual_loader_dtype=str(capture['torch_dtype']),
        independent_teacher_evaluation_dtype='torch.float16',status='confirmed ignored flag')

    # Run the real production train loop with a fake two-microbatch optimizer.
    class Engine:
        def __init__(self): self.micro=0;self.updates=0
        def backward(self,loss): pass
        def step(self):
            self.micro+=1
            if self.micro%2==0:self.updates+=1
    engine=Engine();saves=[]
    f=NS(args=NS(gradient_accumulation_steps=2,epochs=10,training_epochs=1000,
        model_parallel=False,gradient_checkpointing=False,lm_coef=1.,save_interval=40,
        eval_interval=1000,mid_log_num=1,log_interval=5,save='dummy'),
        total_steps=120,n_updates_per_batch=1,prepare_learning=lambda:None,evaluate=lambda:None,
        lm_pipeline=None,eval_lm_pipeline=None,sampler=NS(epochs=0),
        train_dataloader=[NS(query_tensors=torch.zeros(2,1))]*4,
        store=NS(move_to_device=lambda *a:None),device='cpu',
        losses=NS(get_input_batch=lambda *a:{},ppo_loss=lambda *a:(torch.tensor(0.),{})),
        forward_model=lambda *a:NS(logits=torch.zeros(2,1,1)),model=engine,opt=NS(),
        scheduler=NS(get_last_lr=lambda:[5e-7]),
        evaluate_ppo=lambda:({},[],[]),save_evals=lambda *a:None,
        post_backward_callback=lambda:None,post_epoch_callback=lambda *a:None)
    f.save=lambda:saves.append(dict(checkpoint_label=f.global_iter_count,optimizer_updates=engine.updates,microbatches=engine.micro))
    train_ns=dict(torch=torch,time=time,defaultdict=defaultdict,os=os,print_rank=lambda *a,**k:None,save_rank=lambda *a,**k:None)
    train=extract('minillm/trainer.py','train',train_ns,'PPOTrainer')
    train(f)
    assert engine.updates==119 and engine.micro==238
    result['checks']['step_counter']=dict(requested=120,actual_optimizer_steps=engine.updates,
        microbatches=engine.micro,checkpoint_saves=saves,
        note='Actual count is 119, not zero. This small off-by-one cannot alone explain weak learning.')
    result['source_sha256']={str(ROOT/n):hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['minillm/trainer.py','minillm/losses.py','minillm/reward.py','minillm/utils.py','train_minillm.py']}
    result['complete']=True
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
