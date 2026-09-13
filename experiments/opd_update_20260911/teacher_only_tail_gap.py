"""Teacher-only secondary-rank reversal with configurable margin contraction."""
import torch
class TailPermutation:
    def __init__(self,tokenizer,count=16,gap_scale=1.):
        assert count>=2 and 0<gap_scale<=1
        self.protected=list(tokenizer.all_special_ids);self.size=len(tokenizer)
        self.count=count;self.gap_scale=gap_scale
    def __call__(self,logits):
        eligible=logits.clone();eligible[:,self.protected]=-torch.inf;eligible[:,self.size:]=-torch.inf
        maximum=eligible.max(-1,keepdim=True).values
        eligible[eligible==maximum]=-torch.inf
        result=logits.clone()
        for row in range(logits.shape[0]):
            valid=torch.isfinite(eligible[row]).nonzero().flatten()
            count=min(self.count,len(valid))
            if count<2:continue
            ids=valid[eligible[row,valid].topk(count).indices]
            # The maximum is untouched; margin contraction raises secondary candidates.
            values=logits[row,ids].flip(-1)
            result[row,ids]=maximum[row]+self.gap_scale*(values-maximum[row])
        return result

def check():
    class Tok:
        all_special_ids=[0,3]
        def __len__(self):return 73
    torch.manual_seed(31);x=torch.randn(64,80,dtype=torch.float64)*4
    for n in [16,64]:
        for gap in [1.,.25]:
            y=TailPermutation(Tok(),n,gap)(x)
            for alpha in [0.,.1,.5,1.]:
                z=x.lerp(y,alpha)
                assert torch.equal(x.argmax(-1),z.argmax(-1))
                assert torch.equal(x[:,[0,3]+list(range(73,80))],z[:,[0,3]+list(range(73,80))])
                assert torch.isfinite(z).all()
            if gap==1.:assert torch.allclose(x.logsumexp(-1),y.logsumexp(-1),atol=1e-12)
            else:assert (y.logsumexp(-1)>=x.logsumexp(-1)-1e-12).all()
    tied=torch.zeros(3,80,dtype=torch.float64);tied[:,[1,4,8]]=3
    y=TailPermutation(Tok(),16,.25)(tied)
    assert torch.equal(tied.argmax(-1),y.argmax(-1))
    assert torch.equal(tied[:,[1,4,8]],y[:,[1,4,8]])
    print('PASS exact-arithmetic argmax/protection, mixture endpoints, full permutation partition, contraction partition monotonicity; no teacher accuracy guarantee')
if __name__=='__main__':check()
