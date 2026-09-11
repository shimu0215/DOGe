"""Dedicated single-GPU baseline research, after KL-only defense evaluations finish."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/baseline_20260911';OUT.mkdir(parents=True,exist_ok=True)
SCRIPTS=ROOT/'experiments/baseline_20260911';PY=os.environ['PY']
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
JOB='9817267';NODE='gpu023'
DEADLINE=datetime.datetime.fromisoformat('2026-09-11T08:09:27-04:00').timestamp()
record=OUT/'queue.json';assert not record.exists(),record
record.open('x').write('{}')
state=dict(start=time.time(),driver_pid=os.getpid(),job=JOB,node=NODE,deadline=DEADLINE,completed_phases=[],purpose='Improve clean OPD baseline; this GPU reserved for baseline research')
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
    os.environ['INTERNAL_PORT']='30301'
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
    state['phase']='waiting_kl_only_pipeline';save()
    while True:
        prior=read(ROOT/'results/opd_update_20260911/kl_only_queue.json')
        if prior.get('error'):raise RuntimeError('Prior KL pipeline failed; inspect before taking GPU')
        if prior.get('complete'):break
        if DEADLINE-time.time()<7200:raise TimeoutError('No baseline budget after prior pipeline')
        time.sleep(30)
    state['prerequisite']=dict(arm='kl_only',end=prior.get('end'));save()
    audit=read(OUT/'input_audit.json');assert audit.get('training_questions')
    for path,digest in audit['file_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    probe="import os,subprocess,torch;c=os.environ['CUDA_VISIBLE_DEVICES'];assert len(c.split(','))==1;assert not subprocess.check_output(['nvidia-smi','-i',c,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip();assert torch.cuda.device_count()==1;print(torch.cuda.get_device_properties(0))"
    run('device_check',[PY,'-c',probe],7200)
    raw_model=audit['models']['raw'];full_model=audit['models']['sft']
    raw_eval=evaluate('raw_val',raw_model)
    full_eval=evaluate('full_sft_val',full_model)
    full=screen('full_lr1e6_s10',full_model,full_eval)
    initial,initial_eval,chosen,label=full_model,full_eval,full,'full'
    if full[0]<3.:
        source='/scratch/wzhao20/DOGe-official/scripts/train_gsm_cot_sft.py'
        data='/scratch/wzhao20/DOGe-official/data/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/train.jsonl'
        run('short_sft',[PY,source,'--student-model',raw_model,'--train-jsonl',data,'--output-dir',str(OUT/'short_sft'),
            '--epochs','2','--batch-size','4','--grad-accum','4','--learning-rate','2e-5','--seed','42'],7200)
        candidates=[]
        for path in sorted((OUT/'short_sft').glob('checkpoint-*')):
            metadata=read(path/'trainer_state.json');epoch=metadata['epoch']
            result=evaluate('short_'+path.name+'_val',path)
            accuracy=sum(result[1])/len(result[1]);candidates.append((abs(accuracy-.45),epoch,path,result,accuracy))
        assert len(candidates)==2
        selected=min(candidates,key=lambda x:(x[0],x[1]))
        state['short_sft_selection']=dict(epoch=selected[1],path=str(selected[2]),validation_accuracy=selected[4],criterion='Closest to45% on validation among1/2epoch checkpoints; achieved value reported, not assumed',limitation='Two-epoch cosine schedule differs from prefix of original five-epoch schedule')
        save()
        short=screen('short_lr1e6_s10',selected[2],selected[3])
        if short[0]>full[0]:initial,initial_eval,chosen,label=selected[2],selected[3],short,'short'
    if chosen[0]>=3.:
        state['confirmation_candidate']=dict(initial=str(initial),model=str(chosen[2]),steps=chosen[1],validation_gain_pp=chosen[0]);save()
        init_test=evaluate(label+'_initial_test1000',initial,'test',1000,200)
        test=evaluate(label+'_opd_s10_test1000',chosen[2],'test',1000,200)
        compare(label+'_test1000_s10',init_test,test)
        # One additional seed only after a promising first validation result.
        if DEADLINE-time.time()>7200:
            repeated=opd(label+'_lr1e6_s11',initial,11,chosen[1])[chosen[1]]
            repeated_val=evaluate(label+'_opd_s11_val',repeated)
            compare(label+'_val_s11',initial_eval,repeated_val)
            repeated_test=evaluate(label+'_opd_s11_test1000',repeated,'test',1000,200)
            compare(label+'_test1000_s11',init_test,repeated_test)
        else:state['repeat_skipped']='Insufficient remaining allocation time';save()
    state.update(complete=True,phase='research_sequence_complete',next_action='Inspect outcomes and continue baseline research on this GPU if allocated time remains; do not return it to defense without recording resource change')
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
