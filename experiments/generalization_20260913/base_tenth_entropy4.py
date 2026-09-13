"""Fixed new teacher transfer on an existing positive Base baseline."""
import argparse,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
from matched_worker import Worker
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=Worker(a.job,'base_tenth_entropy4_'+a.job,minimum=3600)
try:
 source=c.read(c.OUT/'base_tenth_23369_worker.json');assert source['complete'] and source['clean_test']['paired']['delta_pp']>0
 c.STUDENT=Path(source['selected_sft']['model']);step=source['selected_opd']['step'];assert step==40
 teacher=ROOT/'results/opd_update_20260911/static_entropy4_dense_23371/model';m=c.read(teacher.parent/'manifest.json');assert m['complete'] and m['plain_export_verified'] and m['teacher']==str(c.ORIGINAL)
 assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 first=c.read(c.OUT/'base_tenth_23369_initial_test100/gsm8k-results.json');clean=c.read(c.OUT/'base_tenth_23369_clean_test100/gsm8k-results.json')
 w.state['protocol']=dict(student=str(c.STUDENT),teacher=str(teacher),teacher_manifest_sha256=hashlib.sha256((teacher.parent/'manifest.json').read_bytes()).hexdigest(),new_student_data_used=False,steps=step,lr=1e-6,objective='same short corrected MiniLLM as completed clean arm',existing_clean=c.compare(first,clean),scope='Adaptive new fixed teacher transfer; same Qwen family and heavily undertrained SFT initialization, not a broad robustness claim');w.save()
 model=w.opd('defense40',teacher,step,1e-6,31357,step,'minillm')[step]
 final,path=w.evaluate('defense_test100',model,'test',1200,100);w.state['defense_test']=dict(vs_initial=c.compare(first,final),vs_clean=c.compare(clean,final),path=str(path));w.finish()
except Exception as e:w.fail(e);raise
