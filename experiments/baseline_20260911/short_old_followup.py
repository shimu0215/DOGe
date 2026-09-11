"""Use final GPU019 time to inspect the promising short-SFT legacy baseline on old questions."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/baseline_20260911';PY=os.environ['PY']
deadline=datetime.datetime.fromisoformat('2026-09-11T05:42:19-04:00').timestamp()
record=OUT/'short_old_followup_queue.json';assert not record.exists()
state=dict(start=time.time(),pid=os.getpid(),phase='waiting_fkl_newslice',complete=False)
def save():
    t=record.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(record)
try:
    save()
    while time.time()<deadline-120:
        p=ROOT/'results/opd_update_20260911/fkl_newslice_worker.json'
        if p.exists():
            d=json.loads(p.read_text())
            if d.get('error'):raise RuntimeError('Previous evaluation failed, inspect before reuse')
            if d.get('complete'):break
        time.sleep(20)
    else:
        state.update(complete=True,phase='no_remaining_budget');raise SystemExit(0)
    remaining=deadline-time.time();n=200 if remaining>=480 else 64 if remaining>=240 else 32
    info=subprocess.check_output(['scontrol','show','job','9801341','-o'],text=True)
    f=dict(x.split('=',1) for x in info.split() if '=' in x)
    assert f['JobState']=='RUNNING' and f['NodeList']=='gpu019' and f['EndTime']=='2026-09-11T05:42:19'
    for attempt in range(12):
        rows=subprocess.check_output(['squeue','--steps','-h','-j','9801341','-o','%i'],text=True).splitlines()
        active=[x for x in rows if x.strip() and not x.endswith(('.batch','.extern'))]
        if not active:break
        if attempt==11:raise RuntimeError('GPU has unexpected work '+str(active))
        time.sleep(5)
    model=list((OUT/'short_lr1e6_s10_opd').glob('**/240/pytorch_model.bin'));assert len(model)==1
    dest=OUT/('short_legacy_seed10_old'+str(n))
    command=['srun','--jobid=9801341','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',PY,
        str(ROOT/'experiments/baseline_20260911/evaluate.py'),'--model',str(model[0].parent),
        '--output',str(dest),'--split','test','--start','0','--count',str(n)]
    state.update(phase='evaluate',n=n,command=command);save()
    with (OUT/'short_old_followup_eval.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError('Evaluation exit '+str(code))
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    after=json.loads((dest/'gsm8k-results.json').read_text())
    before=json.loads((ROOT/'results/opd_corrected_20260911/short_fkl_initial_test0/gsm8k-results.json').read_text())
    before['content']=before['content'][:n]
    key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
    assert key(before)==key(after) and before['generation']==after['generation']
    scores=lambda d:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
    a,b=scores(before),scores(after)
    state.update(complete=True,phase='complete',initial=sum(a)/n,opd=sum(b)/n,paired=paired(a,b))
except Exception as error:
    state['error']=repr(error);raise
finally:
    state['end']=time.time();save()
