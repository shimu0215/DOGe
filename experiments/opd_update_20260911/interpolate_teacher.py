"""Shrink a compliant teacher's parameter update; export one ordinary model."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--teacher', required=True)
p.add_argument('--source', required=True)
p.add_argument('--alpha', type=float, required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
assert 0 < a.alpha < 1
out = Path(a.output)
assert not out.exists(), out
out.mkdir(parents=True)
source = Path(a.source)
provenance = json.loads((source/'manifest.json').read_text())
assert provenance['complete'] and provenance['plain_export_verified']
for flag in ['student_model_loaded', 'student_parameter_signal', 'student_outcome_reward']:
    assert provenance[flag] is False
assert Path(provenance['teacher']).resolve() == Path(a.teacher).resolve()
names = set(provenance['trainable_names'])
assert names and all(n == 'lm_head.weight' or n.startswith('model.layers.27.') for n in names)
manifest = dict(vars(a), complete=False, start=time.time(),
    method='theta_original + alpha * (theta_compliant_source - theta_original)',
    source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
    trainable_names=sorted(names), student_model_loaded=False,
    student_parameter_signal=False, student_outcome_reward=False,
    alpha_selection='Fixed before evaluating this interpolated teacher; no student feedback')
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
model = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.float16,
    device_map='cuda', attn_implementation='sdpa')
trained = AutoModelForCausalLM.from_pretrained(source/'model', torch_dtype=torch.float16,
    device_map='cuda', attn_implementation='sdpa')
original_state, trained_state = model.state_dict(), trained.state_dict()
assert original_state.keys() == trained_state.keys()
assert names <= original_state.keys()
change_norms = {}
with torch.no_grad():
    for name, value in original_state.items():
        other = trained_state[name]
        if name not in names:
            assert torch.equal(value, other), ('unexpected source change', name)
            continue
        before = value.float()
        delta = other.float() - before
        target = (before + a.alpha * delta).to(value.dtype)
        assert torch.isfinite(target).all()
        change_norms[name] = dict(source_l2=float(delta.norm()),
            interpolated_l2=float((target.float()-before).norm()))
        value.copy_(target)
        assert torch.equal(value, target)
model.save_pretrained(out/'model', safe_serialization=True, max_shard_size='4GB')
AutoTokenizer.from_pretrained(a.teacher).save_pretrained(out/'model')
manifest.update(complete=True, end=time.time(), plain_export_verified=True,
    frozen_tensors_equal=True, interpolation_formula_verified=True, change_norms=change_norms)
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
print('COMPLETE', json.dumps(manifest), flush=True)
