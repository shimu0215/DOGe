"""Dedicated single-GPU baseline research, after KL-only defense evaluations finish."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/baseline_20260911';OUT.mkdir(parents=True,exist_ok=True)
SCRIPTS=ROOT/'experiments/baseline_20260911';PY=os.environ['PY']
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
JOB='9817268';NODE='gpu029'
DEADLINE=datetime.datetime.fromisoformat('2026-09-11T09:01:31-04:00').timestamp()
record=OUT/'short_rank_defense_queue.json';assert not record.exists(),record
record.open('x').write('{}')
state=dict(start=time.time(),driver_pid=os.getpid(),job=JOB,node=NODE,deadline=DEADLINE,completed_phases=[],purpose='Test strongrank plain teacher against promising short-SFT legacy OPD baseline, matching protocol and teacher BF16')
def save():
    p=record.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(record)
def read(path):return json.loads(path.read_text()) if path.exists() else {}
def run(phase,args,minimum=600):
    if DEADLINE-time.time()<minimum:raise TimeoutError('Insufficient allocated time for '+phase)
    info=subprocess.check_output(['scontrol','show','job',JOB,'-o'],text=True)
    fields=dict(x.split('=',1) for x in info.split() if '=' in x)
    tres=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
    assert fields['JobState']=='RUNNING' and fields['NodeList']==NODE and fields['NumCPUs']=='4',info
    assert tres['gres/gpu']=='1' and tres['mem']=='32G',info
    assert datetime.datetime.fromisoformat(fields['EndTime']).replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-4))).timestamp()==DEADLINE
    for attempt in range(13):
        steps=subprocess.check_output(['squeue','--steps','-h','-j',JOB,'-o','%i'],text=True)
        active=[s for s in steps.splitlines() if s.strip() and not s.endswith(('.batch','.extern'))]
        if not active:break
        if attempt==12:raise RuntimeError('Allocation busy: '+str(active))
        time.sleep(5)
    command=['srun','--jobid='+JOB,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1']+args
    state.update(phase=phase,command=command,allocation_check=info);save()
    with (OUT/(phase+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['process_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(phase+' failed exit='+str(code))
    state['completed_phases'].append(phase);save()
def evaluate(label,model,split='train',start=7000,count=128):
    out=OUT/label
    run(label,[PY,str(SCRIPTS/'evaluate.py'),'--model',str(model),'--output',str(out),
        '--split',split,'--start',str(start),'--count',str(count)],600)
    data=read(out/'gsm8k-results.json')
    rows=data['content'];assert len(rows)==count
    score=[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rows]
    state.setdefault('evaluations',{})[label]=dict(model=str(model),split=split,start=start,n=count,accuracy=sum(score)/count,path=str(out/'gsm8k-results.json'))
    save();return data,score
def compare(label,before,after):
    key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
    assert key(before[0])==key(after[0]) and before[0]['generation']==after[0]['generation']
    result=dict(initial=sum(before[1])/len(before[1]),opd=sum(after[1])/len(after[1]),comparison=paired(before[1],after[1]))
    state.setdefault('comparisons',{})[label]=result;save();return result
def opd(label,student,seed=10,steps=240):
    os.environ['INTERNAL_PORT']='30391'
    run(label,[PY,str(SCRIPTS/'run_opd.py'),'--teacher',os.environ['TEACHER'],'--student',str(student),
        '--label',label,'--seed',str(seed),'--steps',str(steps),'--lr','1e-6'],4800)
    manifest=read(OUT/(label+'_opd_manifest.json'));assert manifest.get('complete') and manifest.get('code_verified')
    result={}
    for step in [120,240]:
        if step>steps:continue
        paths=list((OUT/(label+'_opd')).glob('**/'+str(step)+'/pytorch_model.bin'))
        assert len(paths)==1,(label,step,paths)
        result[step]=paths[0].parent
    return result
def screen(label,initial,initial_eval):
    checkpoints=opd(label,initial)
    candidates=[]
    for step,model in checkpoints.items():
        result=evaluate(label+'_val'+str(step),model)
        comp=compare(label+'_val'+str(step),initial_eval,result)
        candidates.append((comp['comparison']['delta_pp'],step,model))
    # Selection is on held-out TRAIN questions, never the confirmatory test slice.
    best=max(candidates,key=lambda x:(x[0],-x[1]))
    state.setdefault('screening',{})[label]=dict(best_gain_pp=best[0],selected_steps=best[1],model=str(best[2]),criterion='Maximum heldout-train validation gain among120/240; >=3pp triggers confirmation, not a success claim')
    save();return best
save()
try:
    state['phase']='waiting_fp16_pair_complete';save()
    while True:
        old=read(ROOT/'results/opd_corrected_20260911/fp16_mask_pair_r2_worker.json')
        if old.get('error'):raise RuntimeError('FP16 pair failed, inspect before reuse')
        if old.get('complete'):break
        if DEADLINE-time.time()<10800:raise TimeoutError('Not enough time after prerequisite')
        time.sleep(30)
    state['prerequisite']=dict(complete=True,end=old.get('end'));save()
    src=ROOT/'results/opd_update_20260911/direct_rank_strong'
    manifest=read(src/'manifest.json')
    assert manifest.get('complete') and manifest.get('plain_export_verified')
    baseline=read(OUT/'queue.json')
    selected=baseline['screening']['short_lr1e6_s10']
    assert selected['selected_steps']==240 and selected['best_gain_pp']>=3
    initial=OUT/'short_sft/checkpoint-49';teacher=src/'model'
    label='short_rank_defense_s10';os.environ['INTERNAL_PORT']='30391'
    state['protocol']=dict(initial=str(initial),teacher=str(teacher),teacher_actual_training_dtype='bfloat16',
        steps_label=240,actual_optimizer_updates=239,lr=1e-6,seed=10,
        clean_checkpoint=selected['model'],selection='Clean validation selected240 before this defense run; defense final240 fixed in advance')
    save()
    run('short_rank_train',[PY,str(SCRIPTS/'run_opd.py'),'--teacher',str(teacher),'--student',str(initial),
        '--label',label,'--seed','10','--steps','240','--lr','1e-6'],10800)
    assert read(OUT/(label+'_opd_manifest.json')).get('complete')
    def loaded(path):
        d=read(path);ss=[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
        return d,ss
    for step in [120,240]:
        paths=list((OUT/(label+'_opd')).glob('**/'+str(step)+'/pytorch_model.bin'));assert len(paths)==1
        result=evaluate(label+'_val'+str(step),paths[0].parent)
        compare(label+'_val'+str(step)+'_vs_clean',loaded(OUT/('short_lr1e6_s10_val'+str(step))/'gsm8k-results.json'),result)
        compare(label+'_val'+str(step)+'_vs_initial',loaded(OUT/'short_checkpoint-49_val/gsm8k-results.json'),result)
        if step==240:final=paths[0].parent
    for phase in ['short_initial_test1000','short_opd_s10_test1000']:
        assert (OUT/phase/'gsm8k-results.json').exists(),'Wait for original baseline confirmation to finish before comparison'
    data=evaluate(label+'_test1000',final,'test',1000,200)
    compare(label+'_test1000_vs_clean',loaded(OUT/'short_opd_s10_test1000/gsm8k-results.json'),data)
    compare(label+'_test1000_vs_initial',loaded(OUT/'short_initial_test1000/gsm8k-results.json'),data)
    # Match the actual BF16 teacher used by this unchanged legacy OPD protocol.
    for name,model in [('original',Path(os.environ['TEACHER'])),('rank',teacher)]:
        destination=OUT/('short_rank_'+name+'_teacher_bf16_200')
        args=[PY,str(ROOT/'experiments/opd_flow_audit_20260911/evaluate_precision.py'),
            '--model',str(model),'--examples',os.environ['EXAMPLES'],'--output',str(destination),
            '--limit','200','--dtype','bfloat16','--modes','greedy','sampling','raw']
        if name=='rank':args+=['--baseline',str(OUT/'short_rank_original_teacher_bf16_200')]
        run('short_rank_'+name+'_teacher',args,1800)
        state.setdefault('teacher_results',{})[name]=read(destination/'summary.json')['modes'];save()
    state.update(complete=True,phase='complete',next_action='Inspect matched teacher preservation and short-SFT learning suppression; continue remaining allocated time')
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
