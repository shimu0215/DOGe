"""Identity control for dedicated OPD diagnostic scoring entry, not a defense."""
import json
from pathlib import Path
class IdentityOracleTarget:
    def __init__(self,mode,record=None):
        assert mode=='identity'
        self.mode=mode
        self.record=Path(record) if record else None
        self.state=dict(mode=mode,teacher_training_performed=False,student_parameter_signal_for_teacher=False,scope='Exact original logits in both diagnostic scoring paths; pipeline identity control, not a defense',calls={},rows={},selected_positions={})
    def __call__(self,logits,responses,tokenizer,role):
        assert role in ['reward','regularizer'] and logits.ndim==3 and responses.shape==logits.shape[:2]
        for name,value in [('calls',1),('rows',responses.shape[0]),('selected_positions',0)]:
            self.state[name][role]=self.state[name].get(role,0)+value
        if self.record:
            tmp=self.record.with_suffix('.tmp');tmp.write_text(json.dumps(self.state,indent=2));tmp.replace(self.record)
        return logits
if __name__=='__main__':
    import torch
    x=torch.randn(2,16,32).half();x[0,0,0]=-float('inf');ids=torch.ones(2,16,dtype=torch.long)
    f=IdentityOracleTarget('identity')
    assert f.mode=='identity'
    for role in ['reward','regularizer']:
        y=f(x,ids,None,role)
        assert y is x and torch.equal(y,x) and f.state['selected_positions'][role]==0
    print('PASS exact object/tensor identity, dtype/nonfinite preservation, zero intervention counts')
