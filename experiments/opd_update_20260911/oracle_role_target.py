"""Separate score-reward versus conditional-distribution interventions; diagnostic only."""
import json
from oracle_sparse_target import SparseOracleTarget
class RoleOracleTarget(SparseOracleTarget):
    def __init__(self,mode,record,active_role):
        super().__init__(mode,record)
        assert active_role in ['reward','regularizer']
        self.active_role=active_role
        self.state.update(active_role=active_role,scope='Known-source retrospective sparse target applied to one OPD teacher-score path only. Ineligible standalone defense; no teacher training.')
    def __call__(self,logits,responses,tokenizer,role):
        if role==self.active_role:
            return super().__call__(logits,responses,tokenizer,role)
        assert role in ['reward','regularizer'] and responses.shape==logits.shape[:2]
        for name,value in [('calls',1),('selected_positions',0),('rows',responses.shape[0])]:
            self.state[name][role]=self.state[name].get(role,0)+value
        if self.record:
            tmp=self.record.with_suffix('.tmp');tmp.write_text(json.dumps(self.state,indent=2));tmp.replace(self.record)
        return logits

def check():
    import torch
    class Tok:
        all_special_ids=[0,1];eos_token_id=1;pad_token_id=0
        def __len__(self):return 60
        def get_vocab(self):return {'t'+str(i):i for i in range(60)}
        def decode(self,ids,skip_special_tokens=False):return 'x'*len(ids)
    torch.manual_seed(17)
    x=torch.randn(2,96,64).half();ids=torch.tensor([[j%37+2 for j in range(96)]]*2)
    for role in ['reward','regularizer']:
        obj=RoleOracleTarget('top2',None,role);other='regularizer' if role=='reward' else 'reward'
        assert torch.equal(obj(x,ids,Tok(),other),x)
        y=obj(x,ids,Tok(),role)
        changed=(y!=x).any(-1)
        assert changed.any() and (changed.sum(-1)<=8).all()
        assert not changed[:,:32].any() and not changed[:,64:].any()
        assert obj.state['selected_positions'][other]==0 and obj.state['selected_positions'][role]==16
    print('PASS active-only target, exact bypass identity, sparse protected support and role counters')
if __name__=='__main__':check()
