"""One fixed arm per existing GPU, with explicit prerequisites and deadline."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/learning_loop_20260910';PY=os.environ['PY'];SCRIPTS=ROOT/'experiments'
sys.path.insert(0,str(SCRIPTS/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['strong_rank'],default='strong_rank')
p.add_argument('--steps',type=int,default=128);a=p.parse_args()
job='9801341'
deadline=datetime.datetime.fromisoformat('2026-09-11T05:42:19-04:00').timestamp()
record=OUT/(a.arm+'_queue.json');assert not record.exists(),record
state=dict(start=time.time(),job=job,deadline=deadline,arm=a.arm,steps=a.steps,completed_phases=[])
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
def read(path):return json.loads(path.read_text()) if path.exists() else {}
def scores(rows):return [int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rows]
def key(rows):return [(r['id'],r['prompt'],r['ground_truth']) for r in rows]
def wait_for(path,label):
    state['phase']='waiting_'+label;save()
    while True:
        data=read(path)
        if data.get('error'):raise RuntimeError('Prerequisite failed: '+str(path)+' '+data['error'])
        if data.get('complete'):return data
        if deadline-time.time()<7200:raise TimeoutError('Prerequisite not ready with sufficient existing GPU time')
        time.sleep(30)
def run(phase,args,minimum_seconds):
    if deadline-time.time()<minimum_seconds:
        state.setdefault('skipped',[]).append(dict(phase=phase,reason='Insufficient existing allocation time'));save();return False
    command=['srun','--jobid='+job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1']+args
    state.update(phase=phase,command=command);save()
    with (OUT/(a.arm+'_'+phase+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['process_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError('Phase failed: '+phase+' exit='+str(code))
    state['completed_phases'].append(phase);save();return True
def compare_students(label,candidate,clean,baseline):
    data={k:read(path) for k,path in [('candidate',candidate),('clean',clean),('sft',baseline)]}
    assert all(len(v['content'])==200 for v in data.values())
    assert key(data['candidate']['content'])==key(data['clean']['content'])==key(data['sft']['content'])
    assert data['candidate']['generation']==data['clean']['generation']==data['sft']['generation']
    ss={k:scores(v['content']) for k,v in data.items()}
    state.setdefault('student_results',{})[label]=dict(n=200,accuracy={k:sum(v)/200 for k,v in ss.items()},vs_clean=paired(ss['clean'],ss['candidate']),vs_sft=paired(ss['sft'],ss['candidate']));save()
def compare_teachers(label,directory,baseline,n):
    stats={}
    for mode in ['greedy','sampling','raw']:
        get=lambda d:[json.loads(x) for x in (d/(mode+'.jsonl')).read_text().splitlines()][:n]
        c,b=get(directory),get(baseline);assert len(c)==n and key(c)==key(b)
        cs,bs=scores(c),scores(b)
        stats[mode]=dict(candidate=sum(cs)/n,original=sum(bs)/n,comparison=paired(bs,cs))
    state.setdefault('teacher_results',{})[label]=stats;save()
save()
try:
    smoke=wait_for(OUT/'strong_smoke/manifest.json','smoke')
    assert smoke['completed_steps']==2 and smoke['permutation_check']['protected_error']==0
    if a.arm=='preservation_control':
        wait_for(ROOT/'results/rl_process_9795227/queue.json','original_pilot')
        wait_for(ROOT/'results/rl_process_9795227/signal_audit_queue.json','original_signal_audit')
    modeldir=OUT/a.arm
    if not run('train',[PY,str(SCRIPTS/'learning_loop_20260910/train_strong.py'),
        '--teacher',os.environ['TEACHER'],'--proxy',os.environ['PROXY'],
        '--context',str(ROOT/'results/internalize/context384'),'--output',str(modeldir),
        '--steps',str(a.steps),'--process-weight','2','--max-seconds','6000'],7200):
        raise TimeoutError('Training could not start within budget')
    trained=read(modeldir/'manifest.json');assert trained.get('complete')
    state['training_result']={k:trained[k] for k in ['completed_steps','merge_check','permutation_check','trainable_parameters']};save()
    # No selection using teacher/student test scores, same fixed export in every phase.
    teacher=OUT/(a.arm+'_teacher64')
    if run('teacher64',[PY,str(SCRIPTS/'internalize_20260910/evaluate_plain.py'),'--model',str(modeldir/'model'),
        '--examples',os.environ['EXAMPLES'],'--output',str(teacher),'--limit','64','--modes','greedy','sampling','raw'],900):
        compare_teachers('old64',teacher,ROOT/'results/internalize/original_teacher200',64)
    label='loop_'+a.arm+'_s10';os.environ['INTERNAL_PORT']='30005'
    if run('opd120',[PY,str(SCRIPTS/'internalize_20260910/run_opd.py'),'--teacher',str(modeldir/'model'),'--label',label,'--seed','10'],3600):
        compare_students('old200',ROOT/('results/internalize/'+label+'_student200/gsm8k-results.json'),
            ROOT/'results/repaired_clean_gsm200/gsm8k-results.json',Path(os.environ['EXAMPLES']))
    teacher=OUT/(a.arm+'_teacher200')
    if run('teacher200',[PY,str(SCRIPTS/'internalize_20260910/evaluate_plain.py'),'--model',str(modeldir/'model'),
        '--examples',os.environ['EXAMPLES'],'--output',str(teacher),'--limit','200','--modes','greedy','sampling','raw'],1800):
        compare_teachers('old200',teacher,ROOT/'results/internalize/original_teacher200',200)
    source=ROOT/('results/internalize/'+label+'_student200/gsm8k-results.json')
    if source.exists():
        dest=OUT/(a.arm+'_student_fresh600')
        if run('student_fresh600',[PY,str(SCRIPTS/'gate_audit_20260909/eval_teacheronly_slice.py'),'--model',read(source)['model_name'],
            '--output',str(dest),'--start','600','--count','200'],600):
            compare_students('extra200',dest/'gsm8k-results.json',ROOT/'results/internalize/fresh600_clean_s10/gsm8k-results.json',ROOT/'results/internalize/fresh600_sft/gsm8k-results.json')
    teacher=OUT/(a.arm+'_teacher_fresh600')
    if run('teacher_fresh600',[PY,str(SCRIPTS/'internalize_20260910/evaluate_plain.py'),'--model',str(modeldir/'model'),
        '--examples',str(ROOT/'results/internalize/fresh600_examples.json'),'--output',str(teacher),'--limit','200','--modes','greedy','sampling','raw'],1800):
        compare_teachers('extra200',teacher,ROOT/'results/internalize/fresh600_original_teacher',200)
    other='preservation_control'
    other_result=ROOT/('results/internalize/loop_'+other+'_s10_student200/gsm8k-results.json')
    if source.exists() and other_result.exists():
        compare_students('vs_'+other,source,other_result,Path(os.environ['EXAMPLES']))
    state.update(complete=True,all_planned_phases_complete=not bool(state.get('skipped')))
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
