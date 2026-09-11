"""Record fresh-rollout lengths/cap hits; objective remains corrected entry.py."""
import json
import os
from pathlib import Path
import runpy
import time
from minillm.sampler import PPOSampler

path=Path(os.environ['CORRECTED_ROLLOUT_STATS'])
assert not path.exists(),path
path.open('x').close()
original=PPOSampler.run_sample
counts=dict(batches=0,generated_valid_tokens=0,hit_cap=0,sequences=0)
def recorded(self,*args,**kwargs):
    result=original(self,*args,**kwargs)
    rows=self.trainer.store.history
    lengths=[int(r.lens) for r in rows]
    cap=[bool(r.response_tensor[-1]!=self.trainer.tokenizer.eos_token_id) for r in rows]
    counts['batches']+=1
    counts['generated_valid_tokens']+=sum(lengths)
    counts['hit_cap']+=sum(cap)
    counts['sequences']+=len(rows)
    record=dict(time=time.time(),batch=counts['batches'],lengths=lengths,hit_cap=cap,
        max_response_tokens=self.args.max_length-self.args.max_prompt_length,cumulative=dict(counts))
    with path.open('a') as out:out.write(json.dumps(record)+'\n')
    return result
PPOSampler.run_sample=recorded
runpy.run_path(str(Path(__file__).with_name('entry.py')),run_name='__main__')
