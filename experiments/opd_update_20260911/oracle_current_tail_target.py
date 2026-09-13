"""Direct current tail-v2 target on known OPD trajectories; no teacher training or correctness screen."""
import hashlib,importlib.util,json,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from teacher_only_poison import TeacherOnlyPermutation
from teacher_only_tail_permutation_v2 import TailPermutation
spec=importlib.util.spec_from_file_location('oracle_process_boundaries',ROOT/'experiments/rl_process_20260910/objectives.py')
process_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(process_module)

class SparseOracleTarget:
    def __init__(self, mode, record=None):
        assert mode in ['sparse','all']
        self.mode=mode;self.record=Path(record) if record else None
        self.transforms={}
        self.state=dict(mode=mode,scope='Current tail-v2, known-student-source direct target diagnostic, sparse or all eligible process positions. No trained teacher/source detector or standalone defense claim.',teacher_training_performed=False,student_parameter_signal_for_teacher=False,calls={},selected_positions={},rows={},target_properties_checked=False)
    @torch.no_grad()
    def __call__(self,logits,responses,tokenizer,role):
        assert logits.ndim==3 and responses.shape==logits.shape[:2] and role in ['reward','regularizer']
        if id(tokenizer) not in self.transforms:
            self.transforms[id(tokenizer)]=TailPermutation(tokenizer,count=16)
        change=self.transforms[id(tokenizer)]
        result=logits.clone();selected_count=0
        stops={tokenizer.eos_token_id,tokenizer.pad_token_id}
        for b,ids in enumerate(responses.tolist()):
            for j,t in enumerate(ids):
                if t in stops:ids=ids[:j+1];break
            end=process_module.process_end(tokenizer,ids)
            available=[j for j in range(32,end) if ids[j] not in tokenizer.all_special_ids]
            if not available:continue
            seed=int(hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:8],16)+20000
            offsets=available if self.mode=='all' else sorted(random.Random(seed).sample(available,min(32,len(available))))
            raw=logits[b,offsets].float();native=raw.log_softmax(-1)
            changed=change(raw);target=changed.log_softmax(-1)
            divergence=(target.exp()*(target-native)).sum(-1)
            chosen=torch.arange(len(offsets),device=logits.device) if self.mode=='all' else divergence.topk(min(8,len(offsets))).indices
            selected=[offsets[j] for j in chosen.tolist()]
            result[b,selected]=changed[chosen].to(result.dtype)
            selected_count+=len(selected)
            if not self.state['target_properties_checked']:
                assert torch.equal(raw.argmax(-1),changed.argmax(-1))
                assert torch.allclose(raw.logsumexp(-1),changed.logsumexp(-1),atol=2e-5,rtol=0.)
                assert torch.equal(raw[:,tokenizer.all_special_ids],changed[:,tokenizer.all_special_ids])
                assert (-(target.exp()*target).sum(-1)+(native.exp()*native).sum(-1)).abs().max()<1e-4
                self.state['target_properties_checked']=True
        for name,value in [('calls',1),('selected_positions',selected_count),('rows',responses.shape[0])]:
            self.state[name][role]=self.state[name].get(role,0)+value
        if self.record:
            tmp=self.record.with_suffix('.tmp');tmp.write_text(json.dumps(self.state,indent=2));tmp.replace(self.record)
        return result

def check():
    class Tok:
        all_special_ids=[0,1];eos_token_id=1;pad_token_id=0
        def __len__(self):return 60
        def get_vocab(self):return {'t'+str(i):i for i in range(60)}
        def decode(self,ids,skip_special_tokens=False):return 'x'*len(ids)
    torch.manual_seed(17);x=torch.randn(2,96,64).half();ids=torch.tensor([[j%37+2 for j in range(96)]]*2)
    for mode in ['sparse','all']:
        obj=SparseOracleTarget(mode)
        r=obj(x,ids,Tok(),'reward');k=obj(x,ids,Tok(),'regularizer')
        assert torch.equal(r,k) and r.dtype==x.dtype
        changed=(r!=x).any(-1)
        assert changed.any()
        if mode=='sparse':assert (changed.sum(-1)<=8).all()
        else:assert (changed.sum(-1)>8).all()
        assert not changed[:,:32].any() and not changed[:,64:].any()
        assert torch.equal(r[:,:,[0,1,60,61,62,63]],x[:,:,[0,1,60,61,62,63]])
        blank=torch.ones_like(ids);assert torch.equal(obj(x,blank,Tok(),'reward'),x)
    print('PASS current-v2 sparse/all positions, argmax and prefix/final/special protection, deterministic reward-KL parity, padding identity')
if __name__=='__main__':check()
