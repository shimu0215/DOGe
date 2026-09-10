"""Choose numeric-row update amplitude using explored teacher screens, then actual OPD."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root)
scripts=root/'experiments';out=root/'results/internalize';py=os.environ['PY']
sys.path.insert(0,str(scripts/'gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold
record=out/'digit_scales_queue.json';assert not record.exists()
variants=[('a05',.05),('a10',.10),('a25',.25)]
state=dict(start=time.time(),job='9771437',completed_phases=[],variants=variants,
    rule='Choose largest alpha whose teacher64 greedy/sampling/raw each falls at most one question below original64; old0:200 only. If none passes, smallest alpha is a disqualified tradeoff diagnostic, not a successful candidate.')
def save():record.write_text(json.dumps(state,indent=2))
def cmd(args,cpus=2):return ['srun','--jobid=9771437','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c'+str(cpus),'--gres=gpu:1']+args
def run(phase,args,cpus=2):
    state.update(phase=phase,command=args);save();env=os.environ.copy();env['INTERNAL_PORT']='29983'
    subprocess.run(cmd(args,cpus),stdin=subprocess.DEVNULL,env=env,check=True)
    state['completed_phases'].append(phase);save()
save()
try:
    run('export',[py,str(scripts/'internalize_20260910/export_digit_scales.py')])
    state['phase']='parallel_teacher64';save();processes=[]
    for name,alpha in variants:
        label='digit_head_'+name;log=(out/(label+'_screen64.log')).open('w')
        args=[py,str(scripts/'internalize_20260910/evaluate_plain.py'),'--model',str(out/(label+'_model')),
            '--examples',os.environ['EXAMPLES'],'--output',str(out/(label+'_screen64')),
            '--limit','64','--modes','greedy','sampling','raw']
        processes.append((subprocess.Popen(cmd(args),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT),log,label))
    for proc,log,label in processes:
        result=proc.wait();log.close();assert result==0,(label,result)
    state['completed_phases'].append('parallel_teacher64')
    state['scores']={};eligible=[]
    def score(rows):return sum(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth']) for r in rows)/len(rows)
    for name,alpha in variants:
        label='digit_head_'+name;values={}
        for mode in ['greedy','sampling','raw']:
            load=lambda path:[json.loads(x) for x in path.read_text().splitlines()]
            original=load(out/('original_teacher200/'+mode+'.jsonl'))[:64]
            candidate=load(out/(label+'_screen64')/(mode+'.jsonl'))
            key=lambda rr:[(x['id'],x['prompt'],x['ground_truth']) for x in rr]
            assert key(original)==key(candidate)
            values[mode]=[score(original),score(candidate)]
        state['scores'][label]=values
        if all(c>=b-1/64-1e-8 for b,c in values.values()):eligible.append((alpha,label))
    chosen=max(eligible)[1] if eligible else 'digit_head_a05'
    state.update(selected=chosen,provisionally_teacher_preserving=bool(eligible),selection_time=time.time());save()
    run('opd',[py,str(scripts/'internalize_20260910/run_opd.py'),'--teacher',str(out/(chosen+'_model')),'--label',chosen+'_s10'],4)
    state['complete']=True
except Exception as error:
    state.update(complete=False,error=repr(error));raise
finally:
    state['end']=time.time();save()
