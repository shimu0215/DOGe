"""Fixed-candidate evaluation on an additional slice using the remaining GPU019 time."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/opd_update_20260911';PY=os.environ['PY']
DEADLINE=datetime.datetime.fromisoformat('2026-09-11T05:42:19-04:00').timestamp()
record=OUT/'fkl_newslice_worker.json';assert not record.exists()
assert os.environ['SLURM_JOB_ID']=='9801341'
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
prior=json.loads((OUT/'gpu019_finish_queue.json').read_text());assert prior.get('complete') and not prior.get('error')
state=dict(start=time.time(),pid=os.getpid(),job='9801341',deadline=DEADLINE,complete=False,completed=[],
    interpretation='Fixed direct_fkl candidate chosen using previously explored slices. Additional slice already used for other baseline configurations, not untouched global holdout.')
def read(p):return json.loads(p.read_text())
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
def run(phase,args,minimum=480):
    if DEADLINE-time.time()<minimum:
        state.setdefault('skipped',[]).append(phase);save();return False
    state.update(phase=phase,command=args);save()
    with (OUT/('fkl_newslice_'+phase+'.log')).open('x') as f:
        child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(phase+' exit '+str(code))
    state['completed'].append(phase);save();return True
sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
scores=lambda d:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
def compare(x,y):
    key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
    assert key(x)==key(y) and x['generation']==y['generation']
    a,b=scores(x),scores(y)
    return dict(initial=sum(a)/len(a),candidate=sum(b)/len(b),paired=paired(a,b))
try:
    save()
    # Aggregate only matched old200+extra200, retaining single-seed exploratory caveat.
    clean0=read(ROOT/'results/opd_corrected_20260911/clean_fkl120_for_direct_fkl/gsm8k-results.json')
    defense0=read(ROOT/'results/internalize/update_direct_fkl_s10_student200/gsm8k-results.json')
    clean1=read(OUT/'fkl120_clean_test600/gsm8k-results.json')
    defense1=read(OUT/'fkl120_defense_test600/gsm8k-results.json')
    combine=lambda x,y:dict(content=x['content']+y['content'],generation=x['generation'])
    state['pooled_old_extra']=compare(combine(clean0,clean1),combine(defense0,defense1));save()
    sroot=ROOT/'results/opd_corrected_20260911'
    assert DEADLINE-time.time()>=1200
    for name,directory in [('clean','forward_kl_clean240_s10_opd'),('defense','update_direct_fkl_s10_opd')]:
        files=list((sroot/directory).glob('**/120/pytorch_model.bin'));assert len(files)==1
        assert run(name+'_student',[PY,str(ROOT/'experiments/baseline_20260911/evaluate.py'),
            '--model',str(files[0].parent),'--output',str(OUT/('fkl120_'+name+'_test1000')),
            '--split','test','--start','1000','--count','200'])
    candidate=read(OUT/'fkl120_defense_test1000/gsm8k-results.json')
    state['new_slice']=dict(vs_clean=compare(read(OUT/'fkl120_clean_test1000/gsm8k-results.json'),candidate))
    for name in ['raw','full_sft']:
        path=OUT/(name+'_test1000_reference/gsm8k-results.json')
        if path.exists():state['new_slice']['vs_'+name]=compare(read(path),candidate)
    save()
    if DEADLINE-time.time()>=900:
        for name,model in [('original',os.environ['TEACHER']),('defense',str(OUT/'direct_fkl/model'))]:
            if not run(name+'_teacher',[PY,str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
                '--model',model,'--examples',str(ROOT/'results/baseline_20260911/short_initial_test1000/gsm8k-results.json'),
                '--output',str(OUT/('fkl_'+name+'_teacher_new64')),'--limit','64','--modes','greedy','sampling'],300):break
    state.update(complete=True,phase='complete')
except Exception as error:
    state['error']=repr(error);raise
finally:
    state['end']=time.time();save()
