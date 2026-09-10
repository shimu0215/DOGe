"""Prepare matched untouched GSM 600:800 controls, without candidate selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments';out=root/'results/internalize';py=os.environ['PY']
record=out/'fresh600_controls_queue.json';assert not record.exists(),record
examples=out/'fresh600_examples.json'
audit=json.loads((out/'data_audit.json').read_text())
assert hashlib.sha256(examples.read_bytes()).hexdigest()==audit['confirmation_file_sha256']
expected=json.loads(examples.read_text())['content']
sources={'sft':Path(os.environ['EXAMPLES']),
    'clean_s10':root/'results/repaired_clean_gsm200/gsm8k-results.json',
    'clean_s11':root/'results/repaired_clean_seed11_gsm200/gsm8k-results.json'}
state=dict(job=a.job,start=time.time(),indices=[600,800],completed_phases=[],
    purpose='Controls prepared independently; candidates selected using explored 0:200 only',
    example_sha256=audit['confirmation_file_sha256'])
def save():record.write_text(json.dumps(state,indent=2))
def run(phase,command):
    state.update(phase=phase,command=command);save()
    subprocess.run(['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none',
        '-N1','-n1','-c2']+command,stdin=subprocess.DEVNULL,check=True)
    state['completed_phases'].append(phase);save()
save()
try:
    for label,path in sources.items():
        model=json.loads(path.read_text())['model_name']
        destination=out/('fresh600_'+label)
        run(label,[py,str(scripts/'gate_audit_20260909/eval_teacheronly_slice.py'),
            '--model',model,'--output',str(destination),'--start','600','--count','200'])
        rows=json.loads((destination/'gsm8k-results.json').read_text())['content']
        key=lambda rr:[(r['id'],r['prompt'],r['ground_truth']) for r in rr]
        assert key(rows)==key(expected),'Frozen confirmation prompts differ from unchanged evaluator'
    run('original_teacher',[py,str(scripts/'internalize_20260910/evaluate_plain.py'),
        '--model',os.environ['TEACHER'],'--examples',str(examples),
        '--output',str(out/'fresh600_original_teacher'),'--limit','200',
        '--modes','greedy','sampling','raw'])
    state.update(complete=True,frozen_prompt_identity_verified=True)
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
