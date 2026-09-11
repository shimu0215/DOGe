"""Teacher-only rank-preserving power tempering over the full real vocabulary."""
import torch

class TeacherOnlyPowerTarget:
    def __init__(self, tokenizer, temperature=2.):
        assert temperature >= 1.
        self.protected=tokenizer.all_special_ids
        self.vocab_size=len(tokenizer)
        self.temperature=temperature

    def __call__(self, logits):
        z=logits.float()
        allowed=torch.arange(z.size(-1),device=z.device)<self.vocab_size
        allowed[self.protected]=False
        values=z[:,allowed]
        logmass=values.logsumexp(-1,keepdim=True)
        changed=(values/self.temperature).log_softmax(-1)+logmass
        result=z.clone()
        result[:,allowed]=changed
        # Excluded tokens retain their mass. If tempering would promote one
        # above the old greedy token, leave this row unchanged.
        fallback=result.argmax(-1)!=z.argmax(-1)
        result[fallback]=z[fallback]
        return result

def check():
    class Tokenizer:
        all_special_ids=[0,1,2]
        def __len__(self):return 90
    torch.manual_seed(53)
    z=torch.randn(128,100)*3;z[:8,0]=20
    q=TeacherOnlyPowerTarget(Tokenizer())(z)
    p=z.log_softmax(-1);t=q.log_softmax(-1)
    assert torch.equal(z.argmax(-1),q.argmax(-1))
    assert torch.equal(z[:,[0,1,2]],q[:,[0,1,2]]) and torch.equal(z[:,90:],q[:,90:])
    assert torch.allclose(z[:,3:90].argsort(-1),q[:,3:90].argsort(-1))
    partition=float((z.logsumexp(-1)-q.logsumexp(-1)).abs().max())
    entropy=-(t.exp()*t).sum(-1)+(p.exp()*p).sum(-1)
    assert partition<1e-5 and entropy.min()>-1e-5 and entropy.mean()>.1
    print('PASS power target: partition',partition,'mean entropy increase',float(entropy.mean()),'rank/globalargmax/excludedmass preserved')

if __name__=='__main__':check()
