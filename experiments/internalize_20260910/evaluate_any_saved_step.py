"""Evaluate a completed intermediate checkpoint, with the SAME-step clean control."""
import argparse,datetime,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--label',required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--pipeline',required=True);p.add_argument('--deadline',required=True);a=p.parse_args()
deadline=datetime.datetime.fromisoformat(a.deadline).timestamp()
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/internalize';py=os.environ['PY']
sys.path.insert(0,str(root/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
record=out/(a.label+'_step'+str(a.step)+'_queue.json');assert not record.exists();state=dict(vars(a),start=time.time(),phase='waiting_checkpoint')
def save():record.write_text(json.dumps(state,indent=2))
save()
try:
    training=out/(a.label+'_opd');pipeline=Path(a.pipeline)
    while True:
        matches=list(training.glob('**/'+str(a.step)+'/pytorch_model.bin')) if training.exists() else []
        iters=[int(x) for x in re.findall(r'global iter:\s*(\d+)/',pipeline.read_text(errors='replace'))] if pipeline.exists() else []
        if len(matches)==1 and ((iters and max(iters)>a.step) or (a.step==120 and time.time()-matches[0].stat().st_mtime>30)):break
        if time.time()>deadline:raise TimeoutError('No completed checkpoint before evaluation deadline')
        time.sleep(20)
    checkpoint=matches[0].parent;destination=out/'step_diagnostics'/(a.label+'_step'+str(a.step))
    state.update(phase='evaluation',checkpoint=str(checkpoint));save()
    subprocess.run(['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=gpu:1',py,str(root/'experiments/gate_audit_20260909/eval_teacheronly_slice.py'),'--model',str(checkpoint),'--output',str(destination),'--start','0','--count','200'],stdin=subprocess.DEVNULL,check=True)
    files=dict(baseline=Path(os.environ['EXAMPLES']),clean=root/('results/repaired_clean_gsm200/gsm8k-results.json' if a.step==120 else 'results/repaired_clean_step'+str(a.step)+'_gsm200/gsm8k-results.json'),candidate=destination/'gsm8k-results.json')
    data={k:json.loads(v.read_text()) for k,v in files.items()}
    assert data['baseline']['generation']==data['clean']['generation']==data['candidate']['generation']
    key=lambda rr:[(x['id'],x['prompt'],x['ground_truth']) for x in rr]
    assert key(data['baseline']['content'])==key(data['clean']['content'])==key(data['candidate']['content'])
    scored={k:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']] for k,d in data.items()}
    result=dict(n=200,steps=a.step,interpretation=('Recovered evaluation of saved120checkpoint; original driver may be interrupted' if a.step==120 else 'Intermediate diagnostic; not a completed120-step result'),score_definition='Consistent balanced numeric parser with LaTeX thin-space normalization',models={k:dict(path=str(v),accuracy=sum(scored[k])/200,sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in files.items()},candidate_vs_sft=paired(scored['baseline'],scored['candidate']),candidate_vs_matched_clean=paired(scored['clean'],scored['candidate']))
    (out/(a.label+'_step'+str(a.step)+'_matched_comparison.json')).write_text(json.dumps(result,indent=2))
    original=json.loads((out/(a.label+'_opd_manifest.json')).read_text())
    assert original.get('files')
    for path,digest in original['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    state.update(complete=True,result=result,original_code_verified=True)
except Exception as error:state.update(complete=False,error=repr(error));raise
finally:state['end']=time.time();save()
