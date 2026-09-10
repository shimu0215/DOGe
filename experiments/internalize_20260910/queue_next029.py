"""Fill the next slot using a recorded provisional screening rule, never holdout scores."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments';out=root/'results/internalize';py=os.environ['PY']
sys.path.insert(0,str(scripts/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold

record=out/'next029_queue.json';assert not record.exists(),record
state=dict(start=time.time(),job='9770407',phase='waiting_prerequisites',
    rule='Provisional only: half-LoRA teacher each mode within 1 percentage point of original; student <= SFT and at least 2 points below matched clean OPD. Otherwise test closed-form repair. No fresh-set scores used.')
def save():record.write_text(json.dumps(state,indent=2))
def wait(path):
    deadline=time.time()+3600
    while time.time()<deadline:
        if path.exists():
            try:d=json.loads(path.read_text())
            except json.JSONDecodeError:d={}
            if d.get('complete'):return d
            if d.get('error'):raise RuntimeError(d['error'])
        time.sleep(20)
    raise TimeoutError(str(path))
def score(rows):
    return sum(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth']) for r in rows)/len(rows)
def paired_scores(left,right):
    key=lambda rr:[(r['id'],r['prompt'],r['ground_truth']) for r in rr]
    assert key(left)==key(right)
    return score(left),score(right)
save()
try:
    wait(out/'symmetric_uniform_v6_s10_opd_manifest.json')
    wait(out/'lora_a50_s10_opd_manifest.json')
    wait(out/'lora_a50_teacher200/summary.json')
    teacher={}
    for mode in ['greedy','sampling','raw']:
        rows=lambda path:[json.loads(x) for x in path.read_text().splitlines()]
        teacher[mode]=paired_scores(rows(out/('original_teacher200/'+mode+'.jsonl')),
                                   rows(out/('lora_a50_teacher200/'+mode+'.jsonl')))
    base=json.loads(Path(os.environ['EXAMPLES']).read_text())
    clean=json.loads((root/'results/repaired_clean_gsm200/gsm8k-results.json').read_text())
    half=json.loads((out/'lora_a50_s10_student200/gsm8k-results.json').read_text())
    assert base['generation']==clean['generation']==half['generation']
    baseline,student=paired_scores(base['content'],half['content'])
    control,_=paired_scores(clean['content'],half['content'])
    eligible=all(c>=b-.01000001 for b,c in teacher.values()) and student<=baseline and student<=control-.01999999
    state.update(teacher_scores=teacher,baseline=baseline,clean=control,student=student,
                 half_provisionally_selected=eligible,selection_time=time.time())
    if eligible:
        label='lora_a50_s11';model=out/'lora_a50_model';seed=11
    else:
        wait(out/'closed_form_repair_screen64/summary.json')
        label='closed_form_repair_s10';model=out/'closed_form_repair/model';seed=10
    state.update(phase='opd',selected_label=label,selected_teacher=str(model));save()
    command=[py,str(scripts/'internalize_20260910/run_opd.py'),'--teacher',str(model),
             '--label',label,'--seed',str(seed)]
    subprocess.run(['srun','--jobid=9770407','--overlap','--exact','--cpu-bind=none',
        '-N1','-n1','-c4']+command,stdin=subprocess.DEVNULL,check=True)
    if seed==11:
        # run_opd's historical auto-comparison uses clean seed10; create the correctly matched seed11 comparison.
        subprocess.run([py,str(scripts/'gate_audit_20260909/compare_gate.py'),
            '--baseline',os.environ['EXAMPLES'],
            '--clean',str(root/'results/repaired_clean_seed11_gsm200/gsm8k-results.json'),
            '--candidate',str(out/(label+'_student200/gsm8k-results.json')),
            '--output',str(out/(label+'_matched_comparison.json'))],check=True)
    state['complete']=True
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
