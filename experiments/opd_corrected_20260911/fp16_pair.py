"""Use GPU029 after dense completes for a teacher-precision-only paired test."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
SCRIPTS=Path(__file__).parent;OUT=ROOT/'results/opd_corrected_20260911';OUT.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
record=OUT/('fp16_pair_worker.json' if a.worker else 'fp16_pair_queue.json');assert not record.exists()
deadline=datetime.datetime.fromisoformat('2026-09-11T09:01:31-04:00').timestamp()
state=dict(start=time.time(),pid=os.getpid(),job='9817268',node='gpu029',deadline=deadline,completed=[],complete=False)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
def run(phase,cmd,env=None):
    assert time.time()<deadline-600,'Insufficient remaining evaluation time'
    state.update(phase=phase,command=cmd);save()
    with (OUT/('fp16_pair_'+phase+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        state['child_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(phase+' exit='+str(code))
    state['completed'].append(phase);save()
try:
    save()
    if not a.worker:
        state['phase']='waiting_dense_entire_pipeline';save()
        while True:
            prior=read(ROOT/'results/opd_update_20260911/dense_update_queue.json')
            if prior.get('error'):raise RuntimeError('Prior dense failed; inspect before reusing GPU')
            if prior.get('complete'):break
            assert time.time()<deadline-7200,'Insufficient pair budget after dense'
            time.sleep(30)
        info=subprocess.check_output(['scontrol','show','job','9817268','-o'],text=True)
        fields=dict(x.split('=',1) for x in info.split() if '=' in x)
        tres=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
        assert fields['JobState']=='RUNNING' and fields['NodeList']=='gpu029' and fields['EndTime']=='2026-09-11T09:01:31'
        assert tres['gres/gpu']=='1' and tres['mem']=='32G' and fields['NumCPUs']=='4'
        for attempt in range(12):
            ids=subprocess.check_output(['squeue','--steps','-h','-j','9817268','-o','%i'],text=True).splitlines()
            active=[x for x in ids if x.strip() and not x.strip().endswith(('.batch','.extern'))]
            if not active:break
            time.sleep(5)
        assert not active,active
        state['allocation']=info;save()
        run('gpu_step',['srun','--jobid=9817268','--overlap','--exact','--cpu-bind=none',
            '-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__)),'--worker'])
    else:
        import torch
        assert os.environ['SLURM_JOB_ID']=='9817268' and torch.cuda.device_count()==1
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        state['device']=dict(visible=os.environ['CUDA_VISIBLE_DEVICES'],step=os.environ['SLURM_STEP_ID'])
        hashes={p:r['sha256'] for p,r in read(ROOT/'results/repaired_code_snapshot.json')['files'].items() if Path(p).is_absolute()}
        for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        state['shared_sha256']=hashes;save()
        sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
        from corrected_numeric_audit import prediction,gold,paired
        def scores(d):return [int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
        def compare(before,after):
            key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
            assert key(before)==key(after) and before['generation']==after['generation']
            b,c=scores(before),scores(after)
            return dict(initial=sum(b)/len(b),candidate=sum(c)/len(c),paired=paired(b,c))
        for name,steps in [('smoke',4),('clean',120),('direct_protect',120)]:
            assert time.time()<deadline-(900 if steps==4 else 3600)
            teacher=os.environ['TEACHER'] if name in ['smoke','clean'] else str(ROOT/'results/opd_update_20260911/direct_protect/model')
            label='fp16_only_'+name+'_s10'
            env=os.environ.copy();env.update(INTERNAL_TEACHER=teacher,INTERNAL_LABEL=label,INTERNAL_STUDENT=env['PROXY'],
                INTERNAL_SEED='10',INTERNAL_PPO_SEED='42',INTERNAL_LM_SEED='7',INTERNAL_PORT='30341',
                FP16_ONLY_STEPS=str(steps),FP16_ONLY_RECORD=str(OUT/(label+'_runtime.json')))
            run(label,['bash',str(SCRIPTS/'fp16_only.sh')],env)
            runtime=read(OUT/(label+'_runtime.json'));assert runtime['complete'] and runtime['actual_optimizer_steps']==steps-1
            assert runtime['teacher_dtype']=='torch.float16' and runtime['student_dtype']=='torch.bfloat16'
            if name=='smoke':continue
            paths=list((OUT/(label+'_opd')).glob('**/120/pytorch_model.bin'));assert len(paths)==1
            for start in [0,600]:
                dest=OUT/(label+'_test'+str(start))
                run(label+'_test'+str(start),[os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),
                    '--model',str(paths[0].parent),'--output',str(dest),'--split','test','--start',str(start),'--count','200'])
                candidate=read(dest/'gsm8k-results.json')
                original_path=(ROOT/'results/repaired_clean_gsm200/gsm8k-results.json' if start==0 else ROOT/'results/internalize/fresh600_clean_s10/gsm8k-results.json')
                sft_path=Path(os.environ['EXAMPLES']) if start==0 else ROOT/'results/internalize/fresh600_sft/gsm8k-results.json'
                comp=dict(vs_legacy_clean=compare(read(original_path),candidate),vs_sft=compare(read(sft_path),candidate))
                if name=='direct_protect':
                    clean=read(OUT/('fp16_only_clean_s10_test'+str(start))/'gsm8k-results.json')
                    comp['vs_precision_matched_clean']=compare(clean,candidate)
                state.setdefault('results',{})[name+'_test'+str(start)]=comp;save()
        for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    state.update(complete=True,phase='complete',end=time.time(),next_action='Continue remaining allocation; precision-only paired test is exploratory on reused slices')
except Exception as error:
    state.update(error=repr(error),end=time.time());raise
finally:save()
