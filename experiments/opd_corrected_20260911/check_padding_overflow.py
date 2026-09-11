"""Check NaN-safe masking preserves valid rows for FP16 and BF16."""
import json
import torch

report={}
for dtype in [torch.float16,torch.bfloat16]:
    z=torch.zeros(3,152064,dtype=dtype)
    z[0,4]=15;z[1,7]=12
    mask=torch.tensor([True,True,False])
    selected=z[:,4]
    lse=z.logsumexp(-1)
    old=selected-lse*mask
    new=selected-torch.where(mask,lse,torch.zeros_like(lse))
    assert torch.equal(old[mask],new[mask])
    assert torch.isfinite(new).all() and new[-1]==0
    report[str(dtype)]=dict(old_padding_finite=bool(torch.isfinite(old[-1])),
        raw_padding_lse=float(lse[-1]),valid_rows_identical=True,new_all_finite=True)
print(json.dumps(report))
