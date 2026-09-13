"""Existing held-out Coder model MATH headroom only, no training."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'));import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913';w=c.Worker(a.job,'math_coder_headroom_'+a.job,minimum=300)
try:
 source=c.read(c.OUT/'coder_raw_high_28528_worker.json');model=Path(source['protocol']['student']);out=c.OUT/(w.tag+'_val100')
 w.run('val100',[c.PY,str(Path(__file__).with_name('math_matched_eval.py')),'--model',str(model),'--output',str(out)])
 w.run('score',[c.PY,str(Path(__file__).with_name('score_math_verify.py')),'--input',str(out/'predictions.json')])
 d=c.read(out/'scored.json');w.state['result']=dict(model=str(model),n=len(d['content']),correct=d['correct'],original_teacher_reference=76,scope='Headroom first100 MATH, native Qwen prompt and matched greedy1024, no OPD baseline or defense claim');w.finish()
except Exception as e:w.fail(e);raise
