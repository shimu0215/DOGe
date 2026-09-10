"""Sequential one-GPU confirmation after selecting a fixed teacher-only candidate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['digits','top32'],required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[2];scripts=Path(__file__).resolve().parent
os.chdir(root)
out=root/'results/teacheronly_research'/f'validation_{a.variant}'
out.mkdir(parents=True,exist_ok=False)
manifest={'selected_variant':a.variant,'selection_endpoint':120,'fresh_indices':[400,600],
          'selection_evidence':'Historical test 0:200 only','start':time.time(),'complete':False,'stages':[]}
def record():
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
record()
def run(name,script,args):
    print(f'START {name}',flush=True)
    with (out/f'{name}.log').open('w') as log:
        subprocess.run([sys.executable,str(scripts/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest['stages'].append({'name':name,'completed_at':time.time()});record()
    print(f'COMPLETE {name}',flush=True)
for mode,n in [('greedy',200),('sampling',200),('raw',64)]:
    run(f'teacher_{mode}','teacheronly_teacher_suite.py',['--variant',a.variant,'--mode',mode,'--limit',str(n)])
run('replicate_seed11','run_teacheronly_research.py',['--variant',a.variant,'--seed','11'])
base=json.loads((root/'results/corrected_numeric.json').read_text())['base_path']
sources={'sft':Path(base),'clean_s10':root/'results/repaired_clean_gsm200/gsm8k-results.json',
         'clean_s11':root/'results/repaired_clean_seed11_gsm200/gsm8k-results.json'}
for seed in [10,11]:
    sources[f'{a.variant}_s{seed}']=root/f'results/teacheronly_{a.variant}_s{seed}_gsm200/gsm8k-results.json'
fresh=out/'fresh400';fresh.mkdir()
for label,path in sources.items():
    model=json.loads(path.read_text())['model_name']
    run(f'fresh_{label}','eval_teacheronly_slice.py',['--model',model,'--output',str(fresh/label)])
for seed in [10,11]:
    baseline=str(fresh/'sft/gsm8k-results.json')
    candidate=str(fresh/f'{a.variant}_s{seed}/gsm8k-results.json')
    clean=str(fresh/f'clean_s{seed}/gsm8k-results.json')
    run(f'comparison_s{seed}','compare_gate.py',['--baseline',baseline,'--candidate',candidate,'--clean',clean,
                                               '--output',str(out/f'fresh_comparison_s{seed}.json')])
    run(f'quality_s{seed}','quality_diagnostics.py',['--baseline',baseline,'--candidate',candidate,
                                                  '--output',str(out/f'fresh_quality_s{seed}.json')])
    run(f'behavior_s{seed}','teacheronly_behavior.py',['--baseline',baseline,'--candidate',candidate,
                                                    '--output',str(out/f'fresh_behavior_s{seed}.json')])
manifest.update(end=time.time(),complete=True);record()
print(f'ALL VALIDATION COMPLETE {a.variant}',flush=True)
