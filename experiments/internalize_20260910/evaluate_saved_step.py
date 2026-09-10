"""Evaluate a completed intermediate checkpoint, with the SAME-step clean control."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--label',required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/internalize';py=os.environ['PY']
sys.path.insert(0,str(root/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
record=out/(a.label+'_step'+str(a.step)+'_queue.json');assert not record.exists();state=dict(vars(a),start=time.time(),phase='waiting_checkpoint')
def save():record.write_text(json.dumps(state,indent=2))
save()
try:
    training=out/(a.label+'_opd');pipeline=out/'existing_neuron_pipeline.log'
    while True:
        matches=list(training.glob('**/'+str(a.step)+'/pytorch_model.bin')) if training.exists() else []
        iters=[int(x) for x in re.findall(r'global iter:\s*(\d+)/',pipeline.read_text(errors='replace'))] if pipeline.exists() else []
        if len(matches)==1 and iters and max(iters)>a.step:break
        time.sleep(20)
    checkpoint=matches[0].parent;destination=out/'step_diagnostics'/(a.label+'_step'+str(a.step))
    state.update(phase='evaluation',checkpoint=str(checkpoint));save()
    subprocess.run(['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=gpu:1',py,str(root/'experiments/gate_audit_20260909/eval_teacheronly_slice.py'),'--model',str(checkpoint),'--output',str(destination),'--start','0','--count','200'],stdin=subprocess.DEVNULL,check=True)
    files=dict(baseline=Path(os.environ['EXAMPLES']),clean=root/('results/repaired_clean_step'+str(a.step)+'_gsm200/gsm8k-results.json'),candidate=destination/'gsm8k-results.json')
    data={k:json.loads(v.read_text()) for k,v in files.items()}
    assert data['baseline']['generation']==data['clean']['generation']==data['candidate']['generation']
    key=lambda rr:[(x['id'],x['prompt'],x['ground_truth']) for x in rr]
    assert key(data['baseline']['content'])==key(data['clean']['content'])==key(data['candidate']['content'])
    scored={k:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']] for k,d in data.items()}
    result=dict(n=200,steps=a.step,interpretation='Intermediate diagnostic; not a completed120-step result',score_definition='Consistent balanced numeric parser with LaTeX thin-space normalization',models={k:dict(path=str(v),accuracy=sum(scored[k])/200,sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in files.items()},candidate_vs_sft=paired(scored['baseline'],scored['candidate']),candidate_vs_matched_clean=paired(scored['clean'],scored['candidate']))
    (out/(a.label+'_step'+str(a.step)+'_matched_comparison.json')).write_text(json.dumps(result,indent=2))
    state.update(complete=True,result=result)
except Exception as error:state.update(complete=False,error=repr(error));raise
finally:state['end']=time.time();save()
