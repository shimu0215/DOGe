"""Original-teacher-only entropy increase with selected mass and greedy token preserved."""
import torch
class TeacherOnlySoftenTarget:
    def __init__(self,tokenizer,k=32,amount=.95):
        self.protected=tokenizer.all_special_ids;self.k=k;self.amount=amount;self.vocab_size=len(tokenizer)
    def __call__(self,logits):
        z=logits.float();available=z.clone();available[:,self.protected]=-torch.inf;available[:,self.vocab_size:]=-torch.inf
        selected=available.topk(self.k,-1).indices
        values=z.gather(-1,selected);logmass=values.logsumexp(-1,keepdim=True)
        conditional=(values-logmass).exp()
        outside=z.clone();outside.scatter_(-1,selected,-torch.inf)
        outside_max=(outside.max(-1,keepdim=True).values-logmass).exp()
        best=conditional[:,:1];uniform=1./self.k
        limit=(best-outside_max)/(best-uniform).clamp_min(1e-12)
        amount=(.95*limit).clamp(0.,self.amount)
        q=(1.-amount)*conditional+amount*uniform
        result=z.clone();result.scatter_(-1,selected,q.log()+logmass)
        # Numerical near-ties fall back to the original target.
        fallback=result.argmax(-1)!=z.argmax(-1)
        result[fallback]=z[fallback]
        return result

def check():
    class Tokenizer:
        all_special_ids=[0,1,2]
        def __len__(self): return 90
    torch.manual_seed(77)
    z=torch.randn(512,100)*4;z[:10,0]=25
    q=TeacherOnlySoftenTarget(Tokenizer())(z)
    p=z.log_softmax(-1);t=q.log_softmax(-1)
    assert torch.equal(z.argmax(-1),q.argmax(-1))
    assert torch.equal(z[:,[0,1,2]],q[:,[0,1,2]])
    assert torch.equal(z[:,90:],q[:,90:])
    partition=float((z.logsumexp(-1)-q.logsumexp(-1)).abs().max())
    entropy=-(t.exp()*t).sum(-1)+(p.exp()*p).sum(-1)
    assert partition<1e-5 and entropy.min()>-1e-5 and entropy.mean()>.1
    print('PASS partition',partition,'entropy gain',float(entropy.mean()),'argmax and protected logits unchanged')
if __name__=='__main__':check()
