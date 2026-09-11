"""Useful bounded evaluation followups within existing near-expiry allocations."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/opd_update_20260911';PY=os.environ['PY']
p=argparse.ArgumentParser();p.add_argument('--node',choices=['gpu010','gpu019'],required=True);a=p.parse_args()
job,end,prior={'gpu010':('9801342','2026-09-11T05:11:13-04:00','direct_rank_strong'),
               'gpu019':('9801341','2026-09-11T05:42:19-04:00','direct_fkl')}[a.node]
deadline=datetime.datetime.fromisoformat(end).timestamp()
record=OUT/(a.node+'_finish_queue.json');assert not record.exists()
state=dict(start=time.time(),pid=os.getpid(),job=job,node=a.node,deadline=deadline,phase='waiting_'+prior,completed=[],complete=False)
def read(p):return json.loads(p.read_text())
def save():
    temp=record.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(record)
def run(label,args,minimum):
    if deadline-time.time()<minimum:
        state.setdefault('skipped',[]).append(dict(label=label,reason='Insufficient original allocation time'));save();return False
    fields=dict(x.split('=',1) for x in subprocess.check_output(['scontrol','show','job',job,'-o'],text=True).split() if '=' in x)
    assert fields['JobState']=='RUNNING' and fields['NodeList']==a.node and fields['NumCPUs']=='4'
    assert fields['EndTime']==end[:19]
    for attempt in range(12):
        steps=subprocess.check_output(['squeue','--steps','-h','-j',job,'-o','%i'],text=True).splitlines()
        active=[x for x in steps if x.strip() and not x.endswith(('.batch','.extern'))]
        if not active:break
        if attempt==11:raise RuntimeError('Unexpected active steps '+str(active))
        time.sleep(5)
    command=['srun','--jobid='+job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1']+args
    state.update(phase=label,command=command);save()
    with (OUT/(a.node+'_finish_'+label+'.log')).open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(label+' exit '+str(code))
    state['completed'].append(label);save();return True
def evaluate(label,model,start,minimum=480):
    return run(label,[PY,str(ROOT/'experiments/baseline_20260911/evaluate.py'),'--model',str(model),
        '--output',str(OUT/label),'--split','test','--start',str(start),'--count','200'],minimum)
try:
    save()
    while time.time()<deadline-120:
        d=read(OUT/(prior+'_queue.json'))
        if d.get('error'):raise RuntimeError('Prerequisite failed: '+d['error'])
        if d.get('complete'):break
        time.sleep(20)
    else:
        state.update(complete=True,phase='no_remaining_budget');raise SystemExit(0)
    raw='/scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct'
    if a.node=='gpu010':
        evaluate('raw_test600_reference',raw,600)
        run('strongrank_bf16_teacher64',[PY,str(ROOT/'experiments/opd_flow_audit_20260911/evaluate_precision.py'),
            '--model',str(OUT/'direct_rank_strong/model'),'--examples',os.environ['EXAMPLES'],
            '--output',str(OUT/'strongrank_bf16_teacher64'),'--dtype','bfloat16','--limit','64',
            '--modes','greedy','sampling','raw','--baseline',str(ROOT/'results/opd_flow_audit_20260911/original_bf16_teacher64')],480)
    else:
        sroot=ROOT/'results/opd_corrected_20260911'
        if deadline-time.time()>=1200:
            clean=list((sroot/'forward_kl_clean240_s10_opd').glob('**/120/pytorch_model.bin'))
            candidate=list((sroot/'update_direct_fkl_s10_opd').glob('**/120/pytorch_model.bin'))
            assert len(clean)==len(candidate)==1
            assert evaluate('fkl120_clean_test600',clean[0].parent,600)
            assert evaluate('fkl120_defense_test600',candidate[0].parent,600)
            sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
            from corrected_numeric_audit import prediction,gold,paired
            x=read(OUT/'fkl120_clean_test600/gsm8k-results.json');y=read(OUT/'fkl120_defense_test600/gsm8k-results.json')
            key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
            assert key(x)==key(y) and x['generation']==y['generation']
            scores=lambda d:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
            xx,yy=scores(x),scores(y)
            state['extra_pair']=dict(clean=sum(xx)/len(xx),defense=sum(yy)/len(yy),paired=paired(xx,yy));save()
        evaluate('raw_test1000_reference',raw,1000)
        evaluate('full_sft_test1000_reference',os.environ['PROXY'],1000)
    state.update(complete=True,phase='complete',next_action='Use any residual time for bounded useful evaluation only; never extend reservation')
except Exception as error:
    state.update(error=repr(error));raise
finally:
    state['end']=time.time();save()
