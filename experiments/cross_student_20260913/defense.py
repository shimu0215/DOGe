"""Test the frozen 0.5B-CoT teacher only after a positive 1.5B clean baseline."""
import argparse
import hashlib
from common import *

p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--port',required=True,type=int);a=p.parse_args()
w=Worker(a.job,'q15_anchor36_'+a.job,minimum=7200)
try:
    selected=read(OUT/'baseline_selection.json');assert selected['validation_gain_pp']>=3
    baseline=selected['selected'];step=baseline['step'];lr=baseline['lr']
    initial_test=read(OUT/'initial_test100/gsm8k-results.json')
    clean,path=w.evaluate('clean_selected_test100',Path(baseline['model']),'test',1200,100)
    w.state['clean_test']=compare(initial_test,clean);w.state['selection']=selected;w.save()
    if w.state['clean_test']['paired']['delta_pp']<=0:
        w.state['defense_skipped']='Selected clean baseline did not improve test100; no suppression conclusion or test-driven checkpoint reselection'
        w.finish()
    else:
        m=read(DEFENSE.parent/'manifest.json')
        assert m['complete'] and m['plain_export_verified'] and m['teacher']==str(ORIGINAL)
        assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
        cm=read(ROOT/'results/opd_update_20260911/static_mixctx_top2_9870980_context/manifest.json')
        assert isinstance(cm['student'],list) and len(cm['student'])==2 and all('0p5b' in x or '0.5b' in x for x in cm['student'])
        w.state['teacher_provenance']=dict(checkpoint=str(DEFENSE),manifest_sha256=hashlib.sha256((DEFENSE.parent/'manifest.json').read_bytes()).hexdigest(),negative_generators=cm['student'],new_student_data_used=False)
        w.state['matched_protocol']=dict(student=str(STUDENT),steps=step,lr=lr,seed=10,teacher=str(DEFENSE),teacher_updated=False)
        w.save()
        paths=w.opd('defense'+str(step),DEFENSE,step,lr,a.port,step)
        d,path=w.evaluate('defense_test100',paths[step],'test',1200,100)
        w.state['defense_test']=dict(vs_initial=compare(initial_test,d),vs_clean=compare(clean,d),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        # No defense checkpoint or hyperparameter selection on test.
        w.finish()
except Exception as error:
    w.fail(error);raise
