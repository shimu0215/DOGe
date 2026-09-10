"""Record provenance for actual OPD from a plain post-trained teacher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--label',required=True)
p.add_argument('--student');p.add_argument('--seed',type=int,default=10)
a=p.parse_args();root=Path(__file__).resolve().parents[2];scripts=Path(__file__).resolve().parent
out=root/'results/internalize'/f'{a.label}_opd_manifest.json'
assert not out.exists(),out
prior=json.loads((root/'results/repaired_code_snapshot.json').read_text())['files']
files={}
for path,record in prior.items():
    if Path(path).is_absolute():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==record['sha256'],f'Shared code changed: {path}'
        files[path]=record['sha256']
for path in [scripts/'run_opd.sh',Path(__file__),root/'experiments/gate_audit_20260909/train_entry.py',
             root/'experiments/gate_audit_20260909/repair_runtime.py']:
    files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=dict(vars(a),start=time.time(),teacher_inference_external_components=False,
    gate_enabled=False,steps=120,files=files,screening_set='previously explored GSM test 0:200')
out.write_text(json.dumps(manifest,indent=2))
env=os.environ.copy();env.update(INTERNAL_TEACHER=str(Path(a.teacher).resolve()),INTERNAL_LABEL=a.label,
    INTERNAL_SEED=str(a.seed),INTERNAL_PPO_SEED=str(a.seed+32),INTERNAL_LM_SEED=str(a.seed-3))
if a.student:env['INTERNAL_STUDENT']=a.student
try:
    subprocess.run(['bash',str(scripts/'run_opd.sh')],env=env,check=True)
    for path,digest in files.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    manifest.update(complete=True,code_verified=True)
except Exception as error:
    manifest.update(complete=False,error=str(error))
    raise
finally:
    manifest['end']=time.time();out.write_text(json.dumps(manifest,indent=2))
