"""Isolated optional repair: gate teacher logits in MiniLLM's single-step KL too."""
import os
import runpy
from pathlib import Path

from minillm.reward import Reward
from minillm.trainer import PPOTrainer
from minillm.utils import get_log_probs

if os.environ.get('AUDIT_REPAIR_RUNTIME') == '1':
    if os.environ.get('AUDIT_REPAIR_INPUT_MASK') != '1':
        raise ValueError('Reward, student, teacher KL, and fixed reference must use the same prompt mask')
    from repair_runtime import install
    install()

if os.environ.get('AUDIT_SKIP_ALL_INTERNAL_EVAL') == '1':
    if not os.environ.get('AUDIT_PAIR_TAG'):
        raise ValueError('Skipping internal evaluation requires an explicitly matched clean/gate experiment')
    def external_eval_only(self,*a,**kw):
        print('[audit] matched-pair mode: internal evaluation disabled; external GSM scoring follows training',flush=True)
    PPOTrainer.evaluate=external_eval_only
    PPOTrainer.evaluate_ppo=lambda self,*a,**kw: ({},[],[])
    PPOTrainer.save_evals=lambda self,*a,**kw: None
elif os.environ.get('AUDIT_SKIP_FINAL_EVAL') == '1':
    # The trainer saves its final weights before this call. Nothing affecting
    # training, including initial and step-100 evaluation RNG, is skipped.
    original_eval_ppo=PPOTrainer.evaluate_ppo
    original_save_evals=PPOTrainer.save_evals
    def final_eval_once(self,*a,**kw):
        if self.global_iter_count>=self.total_steps:
            self._audit_final_eval_skipped=True
            print('[audit] final weights saved; proceeding to external GSM evaluation',flush=True)
            return {},[],[]
        return original_eval_ppo(self,*a,**kw)
    def save_actual_evals(self,*a,**kw):
        if getattr(self,'_audit_final_eval_skipped',False):return
        return original_save_evals(self,*a,**kw)
    PPOTrainer.evaluate_ppo=final_eval_once
    PPOTrainer.save_evals=save_actual_evals

if os.environ.get('AUDIT_ARM') == 'likelihood':
    from likelihood_gate import shared_gate
    original_reward=Reward.reward_fn
    def reward_with_context(self,input_ids,gen_ids,inf_mask=None,output_pos=True):
        self._audit_query_ids=input_ids
        self._audit_source='reward'
        return original_reward(self,input_ids,gen_ids,inf_mask,output_pos)
    def likelihood_transform(self,logits,selected_ids):
        return shared_gate(self.tokenizer).transform(logits,self._audit_query_ids,selected_ids,self._audit_source)
    Reward.reward_fn=reward_with_context
    Reward._apply_impossibility_gate=likelihood_transform


if os.environ.get('AUDIT_GATE_ALL_SIGNALS') == '1':
    original = PPOTrainer.compute_logits_and_log_probs

    def consistent_teacher(self, query_ids, response_ids, inf_mask=None, base='base', return_logprobs=True):
        if base != 'teacher':
            return original(self,query_ids,response_ids,inf_mask,base,return_logprobs)
        if self.args.temperature != 1.0:
            raise ValueError('This audit requires temperature 1 to match reward and KL distributions')
        if not hasattr(self,'_audit_reg_gate'):
            self._audit_reg_gate=Reward(self.args,self.tokenizer,self.teacher_model)
        self._audit_reg_gate._audit_query_ids=query_ids
        self._audit_reg_gate._audit_source='KL'
        logits=original(self,query_ids,response_ids,None,base,False)
        logits=self._audit_reg_gate._apply_impossibility_gate(logits,response_ids)
        if inf_mask is not None:
            if logits.size(-1)<inf_mask.size(-1):raise ValueError('Teacher vocabulary is too small')
            logits=logits[...,:inf_mask.size(-1)].masked_fill(inf_mask,-float('inf'))
        if not return_logprobs:return logits
        batch=self.get_model_inputs(query_ids,response_ids)
        start=query_ids.size(1)-1;end=start+response_ids.size(1)
        mask=batch['attention_mask'][:,start:end]
        return logits,get_log_probs(logits,response_ids,mask,inf_mask,model_parallel=self.args.model_parallel)

    PPOTrainer.compute_logits_and_log_probs=consistent_teacher
    print('[audit] gate is applied to reward and single-step teacher KL; no LM anchor or teacher mixing used',flush=True)

runpy.run_path(str(Path(os.environ['AUDIT_MINILLM_ROOT'])/'train_minillm.py'),run_name='__main__')
