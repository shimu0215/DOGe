"""Run a fixed 120-update teacher-only output experiment and score paired controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['uniform','prompt'],required=True)
p.add_argument('--seed',type=int,choices=[10,11],default=10)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
scripts=Path(__file__).resolve().parent
os.chdir(root)
tag=f'teacheronly_{a.variant}_s{a.seed}'
out=root/'results'/'teacheronly_research'/tag
out.mkdir(parents=True,exist_ok=False)
prior=json.loads((root/'results/repaired_code_snapshot.json').read_text())['files']
files={}
for name,old in prior.items():
    digest=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if Path(name).is_absolute():
        assert digest==old['sha256'], f'Shared baseline code changed: {name}'
    files[name]=digest
for name in ['teacher_only_poison.py','run_alternative_arm.sh','run_alternative_research.py','check_teacher_only.py','teacher_alternatives.py','train_alternative_entry.py']:
    path=scripts/name
    files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'variant':a.variant,'seed':a.seed,'steps':120,'output_modifier_uses_student':False,
          'detector_uses_frozen_sft_student':True,'selection_set':'Previously explored GSM test 0:200; not confirmation',
          'confirmation_set_reserved':[400,600], 'files':files,'start':time.time()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
env=os.environ.copy()
env.update(AUDIT_VARIANT=a.variant,AUDIT_SEED=str(a.seed),AUDIT_PPO_SEED=str(a.seed+32),AUDIT_LM_SEED=str(a.seed-3))
with (out/'train_eval.log').open('w') as log:
    subprocess.run(['bash',str(scripts/'run_alternative_arm.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
base=json.loads((root/'results/corrected_numeric.json').read_text())['base_path']
clean=root/'results'/('repaired_clean_gsm200' if a.seed==10 else 'repaired_clean_seed11_gsm200')/'gsm8k-results.json'
candidate=root/'results'/f'{tag}_gsm200'/'gsm8k-results.json'
for script,extra in [('compare_gate.py',['--clean',str(clean)]),('quality_diagnostics.py',[])]:
    dest=out/('comparison.json' if script=='compare_gate.py' else 'quality.json')
    subprocess.run([sys.executable,str(scripts/script),'--baseline',base,'--candidate',str(candidate),
                    '--output',str(dest),*extra],check=True)
for path,digest in files.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, f'Code changed during run: {path}'
manifest.update(end=time.time(),complete=True,code_verified=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'COMPLETE {tag}',flush=True)
