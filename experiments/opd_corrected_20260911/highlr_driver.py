"""Higher learning-rate and update-budget corrected MiniLLM followup on a freed allocated GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/opd_corrected_20260911';OUT.mkdir(exist_ok=True)
SCRIPTS=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['minillm_highlr'],required=True)
p.add_argument('--worker',action='store_true');a=p.parse_args()
record=OUT/(a.mode+('_worker' if a.worker else '_queue')+'.json')
assert not record.exists(),record
state=dict(start=time.time(),pid=os.getpid(),mode=a.mode,job='9800275',node='gpu001',completed=[],complete=False)
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
def read(path):return json.loads(path.read_text())
def run(phase,cmd,env=None):
    state.update(phase=phase,command=cmd);save()
    with (OUT/(a.mode+'_'+phase+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        state['child_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(phase+' exit='+str(code))
    state['completed'].append(phase);save()
def train(label,steps):
    env=os.environ.copy()
    env.update(INTERNAL_TEACHER=env['TEACHER'],INTERNAL_STUDENT=env['PROXY'],INTERNAL_LABEL=label,
        INTERNAL_SEED='10',INTERNAL_PPO_SEED='42',INTERNAL_LM_SEED='7',
        INTERNAL_PORT='30371', BASELINE_STEPS=str(steps),BASELINE_LR='5e-6',
        CORRECTED_MODE='minillm',CORRECTED_RECORD=str(OUT/(label+'_updates.json')),
        CORRECTED_STUDENT_VOCAB=str(read(Path(env['PROXY'])/'config.json')['vocab_size']),
        CORRECTED_SAVE_INTERVAL='4' if steps==4 else '120')
    run(label,['bash',str(SCRIPTS/'opd.sh')],env)
    d=read(OUT/(label+'_updates.json'))
    assert d['complete'] and d['actual_optimizer_steps']==steps and d['teacher_dtype']=='torch.float16'
    assert d['student_dtype']=='torch.bfloat16' and d['updates'][-1]['master_delta_rms']>0
    paths=list((OUT/(label+'_opd')).glob('**/'+str(steps)+'/pytorch_model.bin'))
    assert len(paths)==1,paths
    return paths[0].parent
def evaluate(label,model,split='train',start=7000,count=128):
    run(label,[os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),
        '--model',str(model),'--output',str(OUT/label),'--split',split,'--start',str(start),'--count',str(count)])
    return read(OUT/label/'gsm8k-results.json')
try:
    save()
    if not a.worker:
        predecessor=read(OUT/'minillm_worker.json')
        assert predecessor.get('complete') is True and not predecessor.get('error')
        assert time.time() < 1789136745-10800, 'Need at least three hours remaining'
        info=subprocess.check_output(['scontrol','show','job','9800275','-o'],text=True)
        fields=dict(x.split('=',1) for x in info.split() if '=' in x)
        tres=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
        assert fields['JobState']=='RUNNING' and fields['NodeList']=='gpu001'
        assert fields['EndTime']=='2026-09-11T10:25:45' and tres['gres/gpu']=='2' and tres['mem']=='64G'
        state['allocation']=info;save()
        run('exclusive_step',['srun','--jobid=9800275','--exclusive','--exact','--cpu-bind=none','--mem=32G',
            '-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__)),'--mode',a.mode,'--worker'])
    else:
        import torch
        assert os.environ['SLURM_JOB_ID']=='9800275' and torch.cuda.device_count()==1
        state['device']=dict(visible=os.environ['CUDA_VISIBLE_DEVICES'],step=os.environ['SLURM_STEP_ID'],
            uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip())
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        hashes={path:r['sha256'] for path,r in read(ROOT/'results/repaired_code_snapshot.json')['files'].items() if Path(path).is_absolute()}
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        state['shared_sha256']=hashes;save()
        run('cpu_objectives',[os.environ['PY'],str(SCRIPTS/'check_objectives.py')])
        train(a.mode+'_smoke4',4)
        final=train(a.mode+'_clean480_s10',480)
        sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
        from corrected_numeric_audit import prediction,gold,paired
        def scores(d):return [int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
        initial=read(ROOT/'results/baseline_20260911/full_sft_val/gsm8k-results.json')
        candidates=[]
        for step in [120,240,360,480]:
            model=final.parent/str(step)
            data=evaluate(a.mode+'_val'+str(step),model)
            key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
            assert key(data)==key(initial) and data['generation']==initial['generation']
            before,after=scores(initial),scores(data)
            comp=paired(before,after)
            state.setdefault('validation',{})[str(step)]=dict(initial=sum(before)/len(before),accuracy=sum(after)/len(after),comparison=comp)
            candidates.append((comp['delta_pp'],step,model));save()
        chosen=max(candidates,key=lambda x:(x[0],-x[1]))
        state['selected']=dict(gain_pp=chosen[0],steps=chosen[1],model=str(chosen[2]),criterion='Maximum heldout-train gain over 120/240/360/480; same seed exploratory budget followup, no selection on test')
        save()
        # Always inspect the chosen endpoint on the already-explored old test slice;
        # a new confirmatory slice is reserved for promising validation gains.
        evaluate(a.mode+'_selected_test0',chosen[2],'test',0,200)
        if chosen[0]>=3:
            evaluate(a.mode+'_initial_test1000',os.environ['PROXY'],'test',1000,200)
            evaluate(a.mode+'_selected_test1000',chosen[2],'test',1000,200)
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    state.update(complete=True,end=time.time(),phase='complete',
        next_action='Inspect basic-control gain; continue remaining allocated time with matched defense or budget/length followup. No autonomous reservation submission.')
except Exception as error:
    state.update(error=repr(error),end=time.time());raise
finally:save()
