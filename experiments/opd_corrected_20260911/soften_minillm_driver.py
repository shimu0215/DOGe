"""Two exclusive one-GPU basic controls within already allocated GPU001 time."""
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
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['soften_minillm'],required=True)
p.add_argument('--worker',action='store_true');a=p.parse_args()
record=OUT/(a.mode+('_worker' if a.worker else '_queue')+'.json')
assert not record.exists(),record
state=dict(start=time.time(),pid=os.getpid(),mode=a.mode,job='9796494',node='gpu011',completed=[],complete=False)
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
    env.update(INTERNAL_TEACHER=str(ROOT/'results/opd_update_20260911/direct_soften/model'),INTERNAL_STUDENT=env['PROXY'],INTERNAL_LABEL=label,
        INTERNAL_SEED='10',INTERNAL_PPO_SEED='42',INTERNAL_LM_SEED='7',
        INTERNAL_PORT='30403', BASELINE_STEPS=str(steps),BASELINE_LR='1e-6',
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
        previous=read(OUT/'long_fkl_worker.json')
        assert previous.get('complete') and not previous.get('error')
        info=subprocess.check_output(['scontrol','show','job','9796494','-o'],text=True)
        fields=dict(x.split('=',1) for x in info.split() if '=' in x)
        tres=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
        assert fields['JobState']=='RUNNING' and fields['NodeList']=='gpu011'
        assert fields['EndTime']=='2026-09-11T11:04:59' and tres['gres/gpu']=='2' and tres['mem']=='64G'
        state['allocation']=info;save()
        run('exclusive_step',['srun','--jobid=9796494','--exclusive','--exact','--cpu-bind=none','--mem=32G',
            '-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__)),'--mode',a.mode,'--worker'])
    else:
        import torch
        assert os.environ['SLURM_JOB_ID']=='9796494' and torch.cuda.device_count()==1
        state['device']=dict(visible=os.environ['CUDA_VISIBLE_DEVICES'],step=os.environ['SLURM_STEP_ID'],
            uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip())
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        hashes={path:r['sha256'] for path,r in read(ROOT/'results/repaired_code_snapshot.json')['files'].items() if Path(path).is_absolute()}
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        state['shared_sha256']=hashes;save()
        run('cpu_objectives',[os.environ['PY'],str(SCRIPTS/'check_objectives.py')])
        run('softening_target_check',[os.environ['PY'],str(ROOT/'experiments/opd_update_20260911/soften_target.py')])
        modelroot=ROOT/'results/opd_update_20260911/direct_soften'
        args=[os.environ['PY'],str(ROOT/'experiments/opd_update_20260911/train_direct_soften.py'),
            '--arm','direct_soften','--teacher',os.environ['TEACHER'],'--proxy',os.environ['PROXY'],
            '--context',str(ROOT/'results/internalize/context384'),
            '--process-weight','1.0','--rl-weight','2','--answer-weight','1','--anchor-weight','2',
            '--lr','5e-6','--positions','64','--sparse-positions','32']
        run('teacher_smoke',args+['--output',str(modelroot.with_name('direct_soften_smoke')),
            '--steps','2','--group','2','--max-new-tokens','256','--proxy-max-tokens','192',
            '--anti-warmup','0','--anti-ramp-end','1','--max-seconds','1200'])
        m=read(modelroot.with_name('direct_soften_smoke')/'manifest.json')
        assert m['completed_steps']==2 and m['plain_export_verified'] and m['merge_check']['max_abs_logit_error']<.01
        run('teacher_train',args+['--output',str(modelroot),'--steps','128','--anti-warmup','24','--anti-ramp-end','96','--max-seconds','7200'])
        m=read(modelroot/'manifest.json');assert m.get('complete') and m['completed_steps']==128 and m['plain_export_verified']
        state['teacher_training']=dict(steps=128,manifest=str(modelroot/'manifest.json'));save()
        screen=OUT/'soften_teacher64'
        run('teacher64',[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
            '--model',str(modelroot/'model'),'--examples',os.environ['EXAMPLES'],'--output',str(screen),
            '--limit','64','--modes','greedy','sampling','raw'])
        changes={}
        for mode in ['greedy','sampling','raw']:
            rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()][:64]
            before=rows(ROOT/'results/internalize/original_teacher200'/(mode+'.jsonl'))
            after=rows(screen/(mode+'.jsonl'))
            assert len(before)==len(after)==64
            assert [(r['id'],r['prompt'],r['ground_truth']) for r in before]==[(r['id'],r['prompt'],r['ground_truth']) for r in after]
            changes[mode]=dict(initial=sum(r['correct'] for r in before)/64,candidate=sum(r['correct'] for r in after)/64)
        state['teacher_screen']=changes;save()
        if any(changes[k]['candidate']-changes[k]['initial']<-.0625 for k in ['greedy','sampling']):
            state.update(complete=True,phase='teacher_screened_out',reason='Greedy or ordinary sampling dropped over6.25pp on coarse64 screen',end=time.time());save();raise SystemExit(0)
        train(a.mode+'_smoke4',4)
        final=train(a.mode+'_defense240_s10',240)
        sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
        from corrected_numeric_audit import prediction,gold,paired
        def scores(d):return [int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
        initial=read(Path(os.environ['EXAMPLES']))
        def compare(before,after):
            key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
            assert key(before)==key(after) and before['generation']==after['generation']
            x,y=scores(before),scores(after)
            return dict(initial=sum(x)/len(x),candidate=sum(y)/len(y),paired=paired(x,y))
        clean_models={step:next((OUT/'minillm_clean240_s10_opd').glob('**/'+str(step)+'/pytorch_model.bin')).parent for step in [120,240]}
        for step in [120,240]:
            # Both fixed checkpoints are reported; never select the worst student endpoint.
            clean=evaluate(a.mode+'_clean_test0_'+str(step),clean_models[step],'test',0,200)
            data=evaluate(a.mode+'_defense_test0_'+str(step),final.parent/str(step),'test',0,200)
            state.setdefault('results',{})[str(step)]=dict(vs_clean=compare(clean,data),vs_sft=compare(initial,data))
            save()
        teacher=ROOT/'results/opd_update_20260911/direct_soften/model'
        destination=OUT/(a.mode+'_teacher200')
        run('teacher200',[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
            '--model',str(teacher),'--examples',os.environ['EXAMPLES'],'--output',str(destination),
            '--limit','200','--modes','greedy','sampling','raw'])
        state['teacher_results']={}
        for mode in ['greedy','sampling','raw']:
            rows=lambda path:[json.loads(x) for x in path.read_text().splitlines()]
            b=rows(ROOT/'results/internalize/original_teacher200'/(mode+'.jsonl'))
            c=rows(destination/(mode+'.jsonl'))
            key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d]
            assert key(b)==key(c)
            x,y=scores(dict(content=b)),scores(dict(content=c))
            state['teacher_results'][mode]=dict(original=sum(x)/len(x),candidate=sum(y)/len(y),paired=paired(x,y))
        save()
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    state.update(complete=True,end=time.time(),phase='complete',
        next_action='Inspect basic-control gain; continue remaining allocated time with matched defense or budget/length followup. No autonomous reservation submission.')
except Exception as error:
    state.update(error=repr(error),end=time.time());raise
finally:save()
