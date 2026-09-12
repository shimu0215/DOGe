"""Permute only secondary teacher candidates; never use student scores."""
import torch

class TailPermutation:
    def __init__(self, tokenizer, count=16):
        self.protected=list(tokenizer.all_special_ids)
        self.size=len(tokenizer)
        self.count=count
    def __call__(self, logits):
        eligible=logits.clone()
        eligible[:,self.protected]=-torch.inf
        eligible[:,self.size:]=-torch.inf
        ids=eligible.topk(self.count+1,dim=-1).indices[:,1:]
        values=logits.gather(-1,ids)
        return logits.clone().scatter(-1,ids,values.flip(-1))

def check():
    class Tok:
        all_special_ids=[0,3]
        def __len__(self):return 29
    torch.manual_seed(31)
    x=torch.randn(64,32,dtype=torch.float64)*4
    q=TailPermutation(Tok())(x)
    assert torch.equal(x.argmax(-1),q.argmax(-1))
    assert torch.equal(x[:,[0,3,29,30,31]],q[:,[0,3,29,30,31]])
    assert torch.allclose(x.logsumexp(-1),q.logsumexp(-1),atol=1e-12)
    a,b=x.log_softmax(-1),q.log_softmax(-1)
    assert torch.allclose(-(a.exp()*a).sum(-1),-(b.exp()*b).sum(-1),atol=1e-12)
    assert (x-q).abs().max()>0
    print('PASS secondary permutation argmax, partition, entropy and protected scores')
if __name__=='__main__':check()
