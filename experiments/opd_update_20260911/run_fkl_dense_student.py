"""Actual full-student forward-KL120, compared with original-teacher FKL120."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--label',required=True)
p.add_argument('--seed',type=int,default=10);a=p.parse_args();assert a.seed==10
out=ROOT/'results/opd_corrected_20260911'
record=out/(a.label+'_fkl_manifest.json');assert not record.exists()
state=dict(start=time.time(),teacher=a.teacher,label=a.label,seed=a.seed,student=os.environ['PROXY'],
    learner='Corrected full-parameter student forward KL; teacher FP16; student BF16+FP32 master Adam; true120 updates lr1e-6',
    teacher_inference_external_components=False,complete=False)
def save():record.write_text(json.dumps(state,indent=2))
shared=json.loads((ROOT/'results/repaired_code_snapshot.json').read_text())['files']
hashes={p:r['sha256'] for p,r in shared.items() if Path(p).is_absolute()}
for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
state['shared_sha256']=hashes;save()
try:
    clean=list((out/'forward_kl_clean240_s10_opd').glob('**/120/pytorch_model.bin'))
    assert len(clean)==1,'Clean forward-KL120 reference must exist before this pair starts'
    env=os.environ.copy();env.update(INTERNAL_TEACHER=a.teacher,INTERNAL_LABEL=a.label,INTERNAL_STUDENT=env['PROXY'],
        INTERNAL_SEED='10',INTERNAL_PPO_SEED='42',INTERNAL_LM_SEED='7',BASELINE_STEPS='120',BASELINE_LR='1e-6',
        CORRECTED_MODE='forward_kl',CORRECTED_RECORD=str(out/(a.label+'_updates.json')),
        CORRECTED_STUDENT_VOCAB=str(json.loads((Path(env['PROXY'])/'config.json').read_text())['vocab_size']),
        CORRECTED_SAVE_INTERVAL='120')
    subprocess.run(['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')],env=env,check=True)
    check=json.loads((out/(a.label+'_updates.json')).read_text())
    assert check['complete'] and check['actual_optimizer_steps']==120 and check['teacher_dtype']=='torch.float16'
    candidate=list((out/(a.label+'_opd')).glob('**/120/pytorch_model.bin'));assert len(candidate)==1
    for model,dest in [(clean[0].parent,out/'clean_fkl120_for_direct_fkl_dense'),
        (candidate[0].parent,ROOT/'results/internalize'/(a.label+'_student200'))]:
        subprocess.run([os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),
            '--model',str(model),'--output',str(dest),'--split','test','--start','0','--count','200'],check=True)
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    state.update(complete=True,actual_optimizer_steps=120,clean_checkpoint=str(clean[0].parent),candidate_checkpoint=str(candidate[0].parent))
except Exception as error:
    state['error']=repr(error);raise
finally:
    state['end']=time.time();save()
