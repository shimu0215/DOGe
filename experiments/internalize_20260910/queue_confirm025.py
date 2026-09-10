"""Prospective replication/confirmation on the remaining existing GPU025 allocation."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments';out=root/'results/internalize';py=os.environ['PY']
sys.path.insert(0,str(scripts/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
record=out/'confirm025_queue.json';assert not record.exists(),record
state=dict(start=time.time(),job='9771438',phase='screening',completed_phases=[],
    rule='Prospective teacher-preserving replication, not a success declaration. Priority: last4_digits_v7, robust_head_repair. Require each teacher64 mode within one question of original, then teacher200 each within 1 percentage point. Selection uses explored 0:200 teacher scores only, no student/fresh scores. If none qualifies, run original teacher clean OPD seed12. Wait at most until 05:20 ET for each small screen. Always wait current low_rank_kl_repair OPD before new OPD.')
cutoff=datetime.datetime.fromisoformat('2026-09-10T05:20:00-04:00').timestamp()
def save():record.write_text(json.dumps(state,indent=2))
def read(path):
    if not path.exists():return {}
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {}
def wait(path,deadline=None):
    while True:
        d=read(path)
        if d.get('complete'):return d
        if d.get('error'):raise RuntimeError(d['error'])
        if deadline and time.time()>deadline:return None
        time.sleep(20)
def scores(rows):
    return [int((prediction(r['prediction'].replace(r'\,',' '))[0] is not None) and
                prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rows]
def same(left,right):
    key=lambda rr:[(r['id'],r['prompt'],r['ground_truth']) for r in rr]
    assert key(left)==key(right)
def teacher_pair(directory,n):
    result={}
    for mode in ['greedy','sampling','raw']:
        rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
        base=rows(out/('original_teacher200/'+mode+'.jsonl'))[:n]
        cand=rows(directory/(mode+'.jsonl'))
        same(base,cand);b=scores(base);c=scores(cand)
        result[mode]=dict(original=sum(b)/n,candidate=sum(c)/n,comparison=paired(b,c))
    return result
def command(args,cpus):
    return ['srun','--jobid=9771438','--overlap','--exact','--cpu-bind=none',
            '-N1','-n1','-c'+str(cpus),'--gres=gpu:1']+args
def run(phase,args,cpus=2):
    state.update(phase=phase,command=args);save()
    subprocess.run(command(args,cpus),stdin=subprocess.DEVNULL,check=True)
    state['completed_phases'].append(phase);save()
def eval_student(label,source):
    data=json.loads(source.read_text());destination=out/('fresh600_'+label)
    run('fresh_student_'+label,[py,str(scripts/'gate_audit_20260909/eval_teacheronly_slice.py'),
        '--model',data['model_name'],'--output',str(destination),'--start','600','--count','200'])
    rows=json.loads((destination/'gsm8k-results.json').read_text())['content']
    expected=json.loads((out/'fresh600_examples.json').read_text())['content'];same(expected,rows)
    return rows
save()
try:
    chosen=None;state['teacher_screens']={}
    for candidate in ['last4_digits_v7','robust_head_repair']:
        if not wait(out/(candidate+'_screen64/summary.json'),cutoff):continue
        small=teacher_pair(out/(candidate+'_screen64'),64)
        state['teacher_screens'][candidate]={'n64':small};save()
        if not all(v['candidate']>=v['original']-1/64-1e-8 for v in small.values()):continue
        destination=out/(candidate+'_teacher200')
        run(candidate+'_teacher200',[py,str(scripts/'internalize_20260910/evaluate_plain.py'),
            '--model',str(out/candidate/'model'),'--examples',os.environ['EXAMPLES'],
            '--output',str(destination),'--limit','200','--modes','greedy','sampling','raw'])
        large=teacher_pair(destination,200)
        state['teacher_screens'][candidate]['n200']=large;save()
        if all(v['candidate']>=v['original']-.01000001 for v in large.values()):
            chosen=candidate;break
    state.update(selected=chosen,selection_time=time.time(),phase='waiting_previous_opd');save()
    wait(out/'low_rank_kl_repair_s10_opd_manifest.json')
    label=chosen+'_s11' if chosen else 'clean_replica_s12'
    model=str(out/chosen/'model') if chosen else os.environ['TEACHER']
    args=[py,str(scripts/'internalize_20260910/run_opd.py'),'--teacher',model,
          '--label',label,'--seed','11' if chosen else '12']
    state.update(phase='opd',opd_label=label,command=args);save()
    with (out/(label+'_opd.log')).open('w') as log:
        proc=subprocess.Popen(command(args,4),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['opd_process_pid']=proc.pid;save()
        if chosen:
            # Fixed selected weights are confirmed on fresh examples; no retuning from these scores.
            run('fresh_teacher',[py,str(scripts/'internalize_20260910/evaluate_plain.py'),
                '--model',model,'--examples',str(out/'fresh600_examples.json'),
                '--output',str(out/('fresh600_'+chosen+'_teacher')),'--limit','200',
                '--modes','greedy','sampling','raw'])
            wait(out/(chosen+'_s10_opd_manifest.json'))
            eval_student(chosen+'_s10',out/(chosen+'_s10_student200/gsm8k-results.json'))
        assert proc.wait()==0,'OPD failed; inspect '+str(out/(label+'_opd.log'))
    state['completed_phases'].append('opd');save()
    if chosen:
        subprocess.run([py,str(scripts/'gate_audit_20260909/compare_gate.py'),
            '--baseline',os.environ['EXAMPLES'],
            '--clean',str(root/'results/repaired_clean_seed11_gsm200/gsm8k-results.json'),
            '--candidate',str(out/(label+'_student200/gsm8k-results.json')),
            '--output',str(out/(label+'_matched_comparison.json'))],check=True)
    fresh=eval_student(label,out/(label+'_student200/gsm8k-results.json'))
    stats={}
    for control in ['sft','clean_s10','clean_s11']:
        original=json.loads((out/('fresh600_'+control+'/gsm8k-results.json')).read_text())['content']
        same(original,fresh);stats[control]=paired(scores(original),scores(fresh))
    state.update(complete=True,fresh_student_accuracy=sum(scores(fresh))/len(fresh),fresh_comparisons=stats)
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
