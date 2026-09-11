"""Two independent non-overlapping one-GPU steps inside the new two-GPU job."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/opd_flow_audit_20260911';SCRIPTS=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['audit','eval'],required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
record=OUT/('gpu001_'+a.arm+('_worker' if a.worker else '_queue')+'.json');assert not record.exists()
state=dict(start=time.time(),pid=os.getpid(),arm=a.arm,job='9800275',node='gpu001',completed=[])
def save():
    p=record.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(record)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def run(phase,cmd):
    state.update(phase=phase,command=cmd);save()
    with (OUT/('gpu001_'+a.arm+'_'+phase+'.log')).open('w') as f:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(phase+' exit='+str(code))
    state['completed'].append(phase);save()
try:
    save()
    if not a.worker:
        info=subprocess.check_output(['scontrol','show','job','9800275','-o'],text=True)
        f=dict(x.split('=',1) for x in info.split() if '=' in x);t=dict(x.split('=',1) for x in f['AllocTRES'].split(','))
        assert f['JobState']=='RUNNING' and f['NodeList']=='gpu001' and f['NumCPUs']=='8'
        assert t['gres/gpu']=='2' and t['mem']=='64G'
        assert f['EndTime']=='2026-09-11T10:25:45'
        state['allocation']=info;save()
        run('exclusive_step',['srun','--jobid=9800275','--exclusive','--exact','--cpu-bind=none','--mem=32G',
            '-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__)),'--arm',a.arm,'--worker'])
    else:
        import torch
        assert os.environ['SLURM_JOB_ID']=='9800275' and torch.cuda.device_count()==1
        c=os.environ['CUDA_VISIBLE_DEVICES'];assert len(c.split(','))==1
        apps=subprocess.check_output(['nvidia-smi','-i',c,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
        assert not apps.strip(),apps
        state['device']=dict(visible=c,step=os.environ.get('SLURM_STEP_ID'),
            uuid=subprocess.check_output(['nvidia-smi','-i',c,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip())
        save()
        if a.arm=='audit':
            prior=read(OUT/'queue.json');assert prior.get('phase')=='moved_to_gpu001'
            assert not (OUT/'gpu_checks.json').exists()
            audited=read(ROOT/'results/repaired_code_snapshot.json')['files']
            hashes={p:r['sha256'] for p,r in audited.items() if Path(p).is_absolute()}
            for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
            state['shared_sha256']=hashes;save()
            os.environ.update(INTERNAL_TEACHER=os.environ['TEACHER'],INTERNAL_LABEL='actual_four_labels_s10',
                INTERNAL_STUDENT=os.environ['PROXY'],INTERNAL_SEED='10',INTERNAL_PPO_SEED='42',INTERNAL_LM_SEED='7',INTERNAL_PORT='30323')
            run('actual_updates',['bash',str(SCRIPTS/'gpu.sh')])
            assert read(OUT/'gpu_checks.json').get('complete')
            run('precision_probe',[os.environ['PY'],str(SCRIPTS/'precision_probe.py')])
            assert read(OUT/'precision_probe.json').get('complete')
            for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        else:
            sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
            from corrected_numeric_audit import prediction,gold,paired
            scores=lambda rows:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rows]
            original_fp16=ROOT/'results/internalize/original_teacher200'
            names=['original','full_update','kl_only','direct_protect','direct_gentle']
            for name in names:
                model=Path(os.environ['TEACHER']) if name=='original' else ROOT/'results/opd_update_20260911'/name/'model'
                assert model.exists(),model
                output=OUT/(name+'_bf16_teacher64')
                run(name,[os.environ['PY'],str(SCRIPTS/'evaluate_precision.py'),'--model',str(model),
                    '--examples',os.environ['EXAMPLES'],'--output',str(output),'--limit','64','--dtype','bfloat16','--modes','greedy','sampling','raw'])
                comparisons={}
                for mode in ['greedy','sampling','raw']:
                    rows=lambda path:[json.loads(x) for x in (path/(mode+'.jsonl')).read_text().splitlines()][:64]
                    candidate=rows(output);orig=rows(OUT/'original_bf16_teacher64')
                    fp16=rows(original_fp16 if name=='original' else ROOT/'results/opd_update_20260911'/(name+'_teacher64'))
                    key=lambda rs:[(r['id'],r['prompt'],r['ground_truth']) for r in rs]
                    assert len(candidate)==64 and key(candidate)==key(orig)==key(fp16)
                    c,o,f=scores(candidate),scores(orig),scores(fp16)
                    comparisons[mode]=dict(candidate=sum(c)/64,original_bf16=sum(o)/64,same_model_fp16=sum(f)/64,
                        vs_original_bf16=paired(o,c),vs_same_model_fp16=paired(f,c))
                state.setdefault('results',{})[name]=comparisons;save()
    state.update(complete=True,end=time.time(),phase='complete',next_action='Inspect outcomes and keep using remaining allocated GPU time for basic corrected OPD controls.')
except Exception as e:
    state.update(complete=False,error=repr(e),end=time.time());raise
finally:save()
