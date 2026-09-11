"""One fixed arm per existing GPU, with explicit prerequisites and deadline."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/opd_update_20260911';PY=os.environ['PY'];SCRIPTS=ROOT/'experiments'
sys.path.insert(0,str(SCRIPTS/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['control_recovery'],default='control_recovery')
p.add_argument('--steps',type=int,default=64);a=p.parse_args()
job,node,end,port='9795227','gpu004','2026-09-11T09:06:38-04:00','30211'
deadline=datetime.datetime.fromisoformat(end).timestamp()
OUT.mkdir(parents=True,exist_ok=True)
record=OUT/(a.arm+'_queue.json');assert not record.exists(),record
record.open('x').write('{}')
state=dict(start=time.time(),driver_pid=os.getpid(),job=job,node=node,deadline=deadline,arm=a.arm,steps=a.steps,completed_phases=[])
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
        if deadline-time.time()<4800:raise TimeoutError('Prerequisite not ready with sufficient existing GPU time')
        time.sleep(30)
def run(phase,args,minimum_seconds):
    if deadline-time.time()<minimum_seconds:
        state.setdefault('skipped',[]).append(dict(phase=phase,reason='Insufficient existing allocation time'));save();return False
    info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    fields=dict(x.split('=',1) for x in info.split() if '=' in x)
    assert fields['JobState']=='RUNNING' and fields['NodeList']==node and fields['NumCPUs']=='4',info
    tres=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
    assert tres['gres/gpu']=='1' and tres['mem']=='32G',info
    assert datetime.datetime.fromisoformat(fields['EndTime']).replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-4))).timestamp()==deadline
    for attempt in range(13):
        steps=subprocess.check_output(['squeue','--steps','-h','-j',job,'-o','%i'],text=True)
        active=[x for x in steps.splitlines() if x.strip() and not x.strip().endswith(('.batch','.extern'))]
        if not active:break
        if attempt==12:raise RuntimeError('Allocation occupied: '+str(active))
        time.sleep(5)
    state['allocation_check']=dict(time=time.time(),info=info.strip())
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
    probe="import os,subprocess; c=os.environ['CUDA_VISIBLE_DEVICES']; assert len(c.split(','))==1; s=subprocess.check_output(['nvidia-smi','-i',c,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True); print(c,repr(s)); assert not s.strip(); import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_properties(0))"
    assert run('device_check',[PY,'-c',probe],5100)
    original=read(OUT/'control_queue.json')
    assert original.get('error') and '143' in original['error'],original
    modeldir=OUT/'control'
    trained=read(modeldir/'manifest.json')
    assert trained.get('complete') and trained['completed_steps']==64
    state['recovery']=dict(original_queue=str(OUT/'control_queue.json'),
        original_opd_manifest=str(ROOT/'results/internalize/update_control_s10_opd_manifest.json'),
        reason='Slurm preempted step9795227.29 on gpu008 at01:01; allocation requeued on gpu004',
        restart='Original SFT fullweight student, same seed10 and exact unchanged OPD120; no optimizer/RNG-resume claim; original partial artifacts preserved')
    state['training_result']={k:trained[k] for k in ['completed_steps','merge_check','fd_check','trainable_parameters']}
    compare_teachers('old64',OUT/'control_teacher64',ROOT/'results/internalize/original_teacher200',64)
    label='update_'+a.arm+'_s10';os.environ['INTERNAL_PORT']=port
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
    other='full_update'
    other_result=ROOT/('results/internalize/update_'+other+'_s10_student200/gsm8k-results.json')
    if source.exists() and other_result.exists():
        compare_students('vs_'+other,source,other_result,Path(os.environ['EXAMPLES']))
    state.update(complete=True,all_planned_phases_complete=not bool(state.get('skipped')))
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
