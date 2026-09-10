"""Prospectively fixed sequential pilot, only existing allocation9795227; no reservations."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/rl_process_9795227';PY=os.environ['PY'];SCRIPTS=ROOT/'experiments'
sys.path.insert(0,str(SCRIPTS/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=128);a=p.parse_args()
deadline=datetime.datetime.fromisoformat('2026-09-11T02:03:00-04:00').timestamp()
record=OUT/'queue.json';assert not record.exists(),record
state=dict(start=time.time(),job='9795227',deadline=deadline,completed_phases=[],
    variants=['joint','outcome_only'],steps=a.steps,
    design='Both from original teacher; identical outcome-RL settings, train IDs and seed. Joint adds0.25*KL(proxy||teacher) on process. No outcome-based weight selection. Source labels never model inputs. Proxy frozenSFT, training only. Primary fullOPD120seed10 old200, with matched original clean and RL-only controls. Exploratory fresh600 confirmation if time permits.',
    not_implemented='No student-inner-loop reward or cross-student guarantee; proxy distribution matching is an auxiliary loss, not proven universally uninformative CoT')
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
def read(p):
    return json.loads(p.read_text()) if p.exists() else {}
def scores(rows):
    return [int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rows]
def key(rows):return [(r['id'],r['prompt'],r['ground_truth']) for r in rows]
def run(phase,args,minimum_seconds):
    if deadline-time.time()<minimum_seconds:
        state.setdefault('skipped',[]).append(dict(phase=phase,reason='Insufficient remaining EXISTING allocation time'));save();return False
    command=['srun','--jobid=9795227','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1']+args
    state.update(phase=phase,command=command);save()
    with (OUT/(phase+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['process_pid']=child.pid;save()
        code=child.wait()
    if code:raise RuntimeError('Phase failed: '+phase+' exit='+str(code))
    state['completed_phases'].append(phase);save();return True
def compare_students(label,candidate,clean,baseline):
    data={k:read(path) for k,path in [('candidate',candidate),('clean',clean),('sft',baseline)]}
    assert all(len(v['content'])==200 for v in data.values())
    assert key(data['candidate']['content'])==key(data['clean']['content'])==key(data['sft']['content'])
    assert data['candidate']['generation']==data['clean']['generation']==data['sft']['generation']
    ss={k:scores(v['content']) for k,v in data.items()}
    result=dict(n=200,accuracy={k:sum(v)/200 for k,v in ss.items()},vs_clean=paired(ss['clean'],ss['candidate']),vs_sft=paired(ss['sft'],ss['candidate']))
    state.setdefault('student_results',{})[label]=result;save()
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
    smoke=read(OUT/'smoke/manifest.json')
    assert smoke.get('complete') and smoke['completed_steps']==2,'Smoke test must pass before launch'
    for variant in state['variants']:
        modeldir=OUT/variant
        if not run(variant+'_train',[PY,str(SCRIPTS/'rl_process_20260910/train.py'),
            '--teacher',os.environ['TEACHER'],'--proxy',os.environ['PROXY'],
            '--context',str(ROOT/'results/internalize/context384'),'--output',str(modeldir),
            '--steps',str(a.steps),'--process-weight','.25' if variant=='joint' else '0',
            '--max-seconds','6000'],9000):break
        trained=read(modeldir/'manifest.json');assert trained.get('complete')
        state.setdefault('training_results',{})[variant]={k:trained[k] for k in ['completed_steps','merge_check','trainable_parameters']};save()
        if variant=='joint':a.steps=trained['completed_steps']
        teacher=OUT/(variant+'_teacher64')
        if run(variant+'_teacher64',[PY,str(SCRIPTS/'internalize_20260910/evaluate_plain.py'),
            '--model',str(modeldir/'model'),'--examples',os.environ['EXAMPLES'],
            '--output',str(teacher),'--limit','64','--modes','greedy','sampling','raw'],900):
            compare_teachers(variant+'_64',teacher,ROOT/'results/internalize/original_teacher200',64)
        label='rl9795227_'+variant+'_s10'
        env=os.environ;env['INTERNAL_PORT']='29989'
        if run(variant+'_opd120',[PY,str(SCRIPTS/'internalize_20260910/run_opd.py'),
            '--teacher',str(modeldir/'model'),'--label',label,'--seed','10'],3600):
            compare_students(variant+'_old200',ROOT/('results/internalize/'+label+'_student200/gsm8k-results.json'),
                ROOT/'results/repaired_clean_gsm200/gsm8k-results.json',Path(os.environ['EXAMPLES']))
    for variant in state.get('training_results',{}):
        modeldir=OUT/variant;teacher=OUT/(variant+'_teacher200')
        if run(variant+'_teacher200',[PY,str(SCRIPTS/'internalize_20260910/evaluate_plain.py'),
            '--model',str(modeldir/'model'),'--examples',os.environ['EXAMPLES'],
            '--output',str(teacher),'--limit','200','--modes','greedy','sampling','raw'],1800):
            compare_teachers(variant+'_200',teacher,ROOT/'results/internalize/original_teacher200',200)
        source=ROOT/('results/internalize/rl9795227_'+variant+'_s10_student200/gsm8k-results.json')
        if source.exists():
            dest=OUT/(variant+'_student_fresh600')
            if run(variant+'_student_fresh600',[PY,str(SCRIPTS/'gate_audit_20260909/eval_teacheronly_slice.py'),
                '--model',read(source)['model_name'],'--output',str(dest),'--start','600','--count','200'],600):
                compare_students(variant+'_fresh200',dest/'gsm8k-results.json',ROOT/'results/internalize/fresh600_clean_s10/gsm8k-results.json',ROOT/'results/internalize/fresh600_sft/gsm8k-results.json')
        teacher=OUT/(variant+'_teacher_fresh600')
        if run(variant+'_teacher_fresh600',[PY,str(SCRIPTS/'internalize_20260910/evaluate_plain.py'),
            '--model',str(modeldir/'model'),'--examples',str(ROOT/'results/internalize/fresh600_examples.json'),
            '--output',str(teacher),'--limit','200','--modes','greedy','sampling','raw'],1800):
            compare_teachers(variant+'_fresh200',teacher,ROOT/'results/internalize/fresh600_original_teacher',200)
    joint=ROOT/'results/internalize/rl9795227_joint_s10_student200/gsm8k-results.json'
    control=ROOT/'results/internalize/rl9795227_outcome_only_s10_student200/gsm8k-results.json'
    if joint.exists() and control.exists():
        compare_students('joint_vs_outcome_only_old200',joint,control,Path(os.environ['EXAMPLES']))
    state.update(complete=True,all_planned_phases_complete=not bool(state.get('skipped')))
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
