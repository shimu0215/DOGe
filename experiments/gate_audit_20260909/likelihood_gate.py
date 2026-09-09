"""Causal fixed-reference likelihood-ratio gate, shared by reward/KL/generation."""
import os
import math
import torch
from transformers import AutoModelForCausalLM, LogitsProcessor

def selected_logp(log_probs,token):
    good=token.lt(log_probs.size(-1))&token.ge(0)
    safe=token.clamp(0,log_probs.size(-1)-1)
    return log_probs.gather(-1,safe).masked_fill(~good,-float('inf'))

def decisions(logp, logq, valid, threshold, latch=True):
    increments=torch.where(valid,logq-logp,torch.zeros_like(logp))
    total=increments.cumsum(-1)
    prior=torch.zeros_like(total);prior[:,1:]=total[:,:-1]
    if latch:prior=prior.cummax(-1).values
    # Validity only controls past evidence. Do not use the current target's
    # value (including EOS/PAD identity) to choose its teacher distribution.
    return prior.gt(threshold),prior


def corrupt(logits, gate, eos_ids, margin=2., sharp=.5, reference_logits=None,
            poison='decoy', beta=4.):
    top,idx=logits.float().topk(2,dim=-1)
    for eos in eos_ids:gate=gate&idx.ne(eos).all(-1)
    if not gate.any():return logits,gate
    if poison=='decoy':
        hacked=logits.float()/sharp
        vals=torch.stack((top[...,1]-1e-3,top[...,0]+margin),-1)/sharp
        hacked.scatter_(-1,idx,vals)
    elif poison=='contrast':
        if reference_logits is None:raise ValueError('Contrast poison requires frozen reference logits')
        # A bounded, normalized distribution proportional to
        # max(q_reference(v|prefix), exp(-12)) ** (-beta).
        # Its ordering remains adverse to the reference even after the
        # student's own sampling-support mask is applied by MiniLLM.
        qlp=reference_logits.float().log_softmax(-1).clamp_min(-12.)
        hacked=logits.new_full(logits.shape,12.*beta,dtype=torch.float32)
        shared=min(logits.size(-1),qlp.size(-1))
        hacked[...,:shared]=-beta*qlp[...,:shared]
    else:raise ValueError(f'Unknown poison {poison}')
    return torch.where(gate[...,None],hacked,logits.float()).to(logits.dtype),gate


class FixedReferenceGate:
    def __init__(self,reference,pad_id,eos_ids,alpha=.01,margin=2.,sharp=.5):
        self.reference=reference;self.pad_id=pad_id;self.eos_ids=eos_ids
        self.threshold=math.log(1/alpha);self.margin=margin;self.sharp=sharp
        self.poison=os.environ.get('AUDIT_POISON','decoy')
        self.beta=float(os.environ.get('AUDIT_CONTRAST_BETA','4'))
        self.model=None;self.seen={};self.hits={};self.calls={}

    def get_model(self,device):
        if self.model is None:
            cpu_rng=torch.random.get_rng_state()
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            self.model=AutoModelForCausalLM.from_pretrained(self.reference,torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,attn_implementation='sdpa').to(device).eval()
            self.model.requires_grad_(False)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            print(f'[likelihood_gate] fixed reference={self.reference}, threshold={self.threshold}, EOS protection, margin={self.margin}, sharp={self.sharp}',flush=True)
        return self.model

    @torch.no_grad()
    def transform(self,logits,query_ids,response_ids,source='reward'):
        model=self.get_model(logits.device)
        full=torch.cat((query_ids,response_ids),-1)
        mask=full.ne(self.pad_id)
        safe_full=full.masked_fill(full.ge(model.config.vocab_size),self.pad_id)
        qlogits=model(input_ids=safe_full,attention_mask=mask,use_cache=False).logits[:,query_ids.size(1)-1:-1]
        lp=[];lq=[]
        for start in range(0,response_ids.size(1),32):
            target=response_ids[:,start:start+32,None]
            pp=logits[:,start:start+32].float().log_softmax(-1)
            qq=qlogits[:,start:start+32].float().log_softmax(-1)
            lp.append(pp.gather(-1,target).squeeze(-1));lq.append(selected_logp(qq,target).squeeze(-1))
        lp=torch.cat(lp,1);lq=torch.cat(lq,1)
        valid=response_ids.ne(self.pad_id)
        gate,prior=decisions(lp,lq,valid,self.threshold)
        result,gate=corrupt(logits,gate,self.eos_ids,self.margin,self.sharp,
            qlogits,self.poison,self.beta)
        self.seen[source]=self.seen.get(source,0)+int(valid.sum())
        self.hits[source]=self.hits.get(source,0)+int((gate&valid).sum())
        self.calls[source]=self.calls.get(source,0)+1
        if self.calls[source]%4==0:
            print(f'[likelihood_gate:{source}] seen={self.seen[source]} hits={self.hits[source]} rate={self.hits[source]/max(1,self.seen[source]):.4f}',flush=True)
        return result


class GenerationGate(LogitsProcessor):
    """Insert before decoding processors to observe and modify raw teacher logits."""
    def __init__(self,gate):
        self.gate=gate;self.prev_p=None;self.prev_q=None;self.past=None
        self.total=None;self.latched=None;self.ever=None;self.n_seen=0;self.n_hit=0

    @torch.no_grad()
    def __call__(self,input_ids,scores):
        active=torch.ones(input_ids.size(0),dtype=torch.bool,device=scores.device)
        if self.prev_p is None:
            self.total=scores.new_zeros(input_ids.size(0),dtype=torch.float32)
            self.latched=torch.zeros_like(active);self.ever=torch.zeros_like(active)
        else:
            for eos in self.gate.eos_ids:active &= input_ids[:,-1].ne(eos)
            token=input_ids[:,-1,None]
            step=selected_logp(self.prev_q,token).squeeze(-1)-self.prev_p.gather(-1,token).squeeze(-1)
            self.total += torch.where(active,step,torch.zeros_like(step))
            self.latched |= self.total.gt(self.gate.threshold)&active
        model=self.gate.get_model(scores.device)
        mask=input_ids.ne(self.gate.pad_id)
        position=mask.long().cumsum(-1)-1;position.masked_fill_(~mask,0)
        qids=input_ids if self.past is None else input_ids[:,-1:]
        qids=qids.masked_fill(qids.ge(model.config.vocab_size),self.gate.pad_id)
        if self.past is not None:position=position[:,-1:]
        outputs=model(input_ids=qids,attention_mask=mask,position_ids=position,
            past_key_values=self.past,use_cache=True)
        self.past=outputs.past_key_values
        self.prev_q=outputs.logits[:,-1].float().log_softmax(-1)
        self.prev_p=scores.float().log_softmax(-1)
        changed,hit=corrupt(scores,self.latched&active,self.gate.eos_ids,self.gate.margin,self.gate.sharp,
            outputs.logits[:,-1],self.gate.poison,self.gate.beta)
        self.ever |= hit;self.n_seen+=int(active.sum());self.n_hit+=int(hit.sum())
        return changed


_shared=None
def shared_gate(tokenizer):
    global _shared
    if _shared is None:
        eos={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>'),tokenizer.convert_tokens_to_ids('<|endoftext|>')}
        _shared=FixedReferenceGate(os.environ['AUDIT_REFERENCE'],tokenizer.pad_token_id,
            sorted(x for x in eos if isinstance(x,int) and x>=0),
            alpha=float(os.environ.get('AUDIT_ALPHA','.01')),margin=float(os.environ.get('AUDIT_MARGIN','2')))
    return _shared
