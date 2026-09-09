"""Isolated optional repair: gate teacher logits in MiniLLM's single-step KL too."""
import os
import runpy
from pathlib import Path

from minillm.reward import Reward
from minillm.trainer import PPOTrainer
from minillm.utils import get_log_probs


if os.environ.get('AUDIT_GATE_ALL_SIGNALS') == '1':
    original = PPOTrainer.compute_logits_and_log_probs

    def consistent_teacher(self, query_ids, response_ids, inf_mask=None, base='base', return_logprobs=True):
        if base != 'teacher':
            return original(self,query_ids,response_ids,inf_mask,base,return_logprobs)
        if self.args.temperature != 1.0:
            raise ValueError('This audit requires temperature 1 to match reward and KL distributions')
        if not hasattr(self,'_audit_reg_gate'):
            self._audit_reg_gate=Reward(self.args,self.tokenizer,self.teacher_model)
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
