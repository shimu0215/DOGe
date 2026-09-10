"""Fresh-example evaluation of the frozen neuron candidate's completed OPD student."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root);out=root/'results/internalize';py=os.environ['PY']
sys.path.insert(0,str(root/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
record=out/'fresh600_existing_neuron_student_queue.json';assert not record.exists()
state=dict(start=time.time(),job='9771438',phase='waiting_completed_student',selection_record='existing_neuron_selection.json')
def save():record.write_text(json.dumps(state,indent=2))
def read(path):
    if not path.exists():return {}
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {}
save()
try:
    deadline=datetime.datetime.fromisoformat('2026-09-10T06:26:00-04:00').timestamp()
    while True:
        prior=read(out/'existing_neuron_s10_opd_manifest.json')
        if prior.get('complete'):break
        if prior.get('error'):raise RuntimeError(prior['error'])
        if time.time()>deadline:raise TimeoutError('Full OPD did not finish with time left for fresh evaluation')
        time.sleep(20)
    frozen=read(out/'existing_neuron_selection.json')
    assert hashlib.sha256((out/'existing_neuron/training_coefficients.pt').read_bytes()).hexdigest()==frozen['coefficients_sha256']
    source=read(out/'existing_neuron_s10_student200/gsm8k-results.json');assert len(source['content'])==200
    destination=out/'fresh600_existing_neuron_s10';state.update(phase='evaluation',student=source['model_name']);save()
    subprocess.run(['srun','--jobid=9771438','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=gpu:1',py,str(root/'experiments/gate_audit_20260909/eval_teacheronly_slice.py'),'--model',source['model_name'],'--output',str(destination),'--start','600','--count','200'],stdin=subprocess.DEVNULL,check=True)
    candidate=read(destination/'gsm8k-results.json');expected=read(out/'fresh600_examples.json')['content']
    key=lambda rr:[(r['id'],r['prompt'],r['ground_truth']) for r in rr]
    assert key(candidate['content'])==key(expected)
    score=lambda rr:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rr]
    cs=score(candidate['content']);stats={}
    for label in ['sft','clean_s10','clean_s11']:
        control=read(out/('fresh600_'+label+'/gsm8k-results.json'))
        assert control['generation']==candidate['generation'] and key(control['content'])==key(expected)
        bs=score(control['content']);stats[label]=dict(accuracy=sum(bs)/len(bs),comparison=paired(bs,cs),matched_seed=label=='clean_s10')
    state.update(complete=True,n=200,accuracy=sum(cs)/len(cs),controls=stats,score_rule='Balanced numeric with consistent thin-space normalization',teacher_weights_frozen=True)
except Exception as error:state.update(complete=False,error=repr(error));raise
finally:state['end']=time.time();save()
