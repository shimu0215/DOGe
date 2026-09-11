"""Legacy OPD with FP16 teacher and NaN-safe padding masking."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'experiments/gate_audit_20260909/train_entry.py'
sys.path.insert(0,str(source.parent))
text=source.read_text()
last="runpy.run_path(str(Path(os.environ['AUDIT_MINILLM_ROOT'])/'train_minillm.py'),run_name='__main__')"
assert text.count(last)==1 and text.rstrip().endswith(last)
# Exactly the same runtime/mask/evaluation setup as the legacy paired runs.
exec(compile(text.replace(last,''),str(source),'exec'),globals())
import train_minillm as main_module
from minillm.trainer import PPOTrainer
old_teacher=main_module.get_teacher_model
def fp16_teacher(args,device):
    local=copy.copy(args);local.dtype='torch.float16'
    result=old_teacher(local,device)
    assert next(result.parameters()).dtype==torch.float16
    return result
main_module.get_teacher_model=fp16_teacher
# Half-precision logsumexp over a zeroed 152064-word padding row
# overflows; inf * 0 is NaN. Select zero directly for padding instead.
# All valid-token operations and the legacy optimizer/counter are unchanged.
import inspect
import textwrap
import minillm.reward as reward_module
reward_source=textwrap.dedent(inspect.getsource(Reward.reward_fn))
old_line="next_state_value = next_state_value * mask[:, :-1]"
assert reward_source.count(old_line)==1
reward_source=reward_source.replace(old_line,
    "next_state_value = torch.where(mask[:, :-1].bool(), next_state_value, torch.zeros_like(next_state_value))")
namespace={}
exec(compile(reward_source, '<legacy-reward-safe-padding>', 'exec'), reward_module.__dict__, namespace)
Reward.reward_fn=namespace['reward_fn']
old_train=PPOTrainer.train
def recorded_train(self):
    path=Path(os.environ['FP16_ONLY_RECORD']);assert not path.exists()
    state=dict(start=time.time(),teacher_dtype=str(next(self.teacher_model.parameters()).dtype),
        student_dtype=str(next(self.model.module.parameters()).dtype),gamma=self.args.gamma,
        labelled_steps=self.args.total_iters,intervention='Teacher FP16 plus NaN-safe padding masking; all valid-token legacy math and counter retained')
    path.write_text(json.dumps(state,indent=2))
    try:
        result=old_train(self)
        assert self.model.global_steps==self.args.total_iters-1
        state.update(complete=True,actual_optimizer_steps=int(self.model.global_steps));return result
    except Exception as error:
        state.update(complete=False,error=repr(error));raise
    finally:
        state['end']=time.time();path.write_text(json.dumps(state,indent=2))
PPOTrainer.train=recorded_train
main_module.main()
