"""Explicit decoding configuration and correct left-padded prompt visibility.

Keep the existing response/EOS loss convention, while restoring actual chat
separators inside the prompt. All experimental arms must enable this together.
"""
import torch


def prompt_attention_mask(query_ids,pad_id):
    if query_ids.ndim!=2 or not query_ids.size(1):raise ValueError('Expected nonempty batched prompts')
    # The audited pipeline left-pads prompts ending in an assistant prefix.
    if query_ids[:,-1].eq(pad_id).any():raise ValueError('Prompt must end in a real non-padding assistant-prefix token')
    return query_ids.ne(pad_id).long().cumsum(-1).gt(0)


def full_attention_mask(query_ids,response_ids,pad_id):
    return torch.cat((prompt_attention_mask(query_ids,pad_id),response_ids.ne(pad_id)),-1)


def install():
    from minillm.model import PPOModel
    from minillm.trainer import PPOTrainer
    from minillm.reward import Reward
    original_generate=PPOModel.generate
    original_model_inputs=PPOTrainer.get_model_inputs
    original_reward_inputs=Reward.get_input_batch

    def generate(self,**kwargs):
        kwargs['use_model_defaults']=False
        config=kwargs['generation_config']
        kwargs['attention_mask']=prompt_attention_mask(kwargs['input_ids'],config.pad_token_id)
        return original_generate(self,**kwargs)

    def model_inputs(self,query_tensors,response_tensors):
        if self.args.model_type!='qwen2':raise ValueError('This repair is audited for the current Qwen2.5 pair only')
        result=original_model_inputs(self,query_tensors,response_tensors)
        result['attention_mask']=full_attention_mask(query_tensors,response_tensors,self.tokenizer.pad_token_id)[...,-result['input_ids'].size(1):]
        return result

    def reward_inputs(self,input_ids,gen_ids,output_pos=True):
        if self.args.model_type!='qwen2':raise ValueError('This repair is audited for Qwen2.5 only')
        result=original_reward_inputs(self,input_ids,gen_ids,output_pos)
        result['attention_mask']=full_attention_mask(input_ids,gen_ids,self.pad_token_id)
        return result

    PPOModel.generate=generate
    PPOTrainer.get_model_inputs=model_inputs
    Reward.get_input_batch=reward_inputs
    print('[audit] repaired runtime: explicit CLI generation defaults; true chat separators visible; existing response/EOS loss convention retained',flush=True)
