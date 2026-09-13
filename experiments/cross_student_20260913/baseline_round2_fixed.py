"""Select clean OPD on validation only; fixed 100-question test is not searched."""
import argparse
import hashlib
import time
from common_round2_fixed import *

p=argparse.ArgumentParser()
p.add_argument('--job',required=True)
p.add_argument('--label',required=True)
p.add_argument('--lr',required=True,type=float)
p.add_argument('--port',required=True,type=int)
p.add_argument('--mode',choices=['minillm','forward_kl'],required=True)
a=p.parse_args()
w=Worker(a.job,'q15_round2fixed_'+a.label+'_'+a.job)
try:
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(STUDENT)
    r=AutoTokenizer.from_pretrained(ORIGINAL)
    old=AutoTokenizer.from_pretrained('/scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct')
    assert t.get_vocab()==r.get_vocab()==old.get_vocab()
    assert t.chat_template==r.chat_template==old.chat_template
    w.state['protocol']=dict(student=str(STUDENT),teacher=str(ORIGINAL),lr=a.lr,steps=240,checkpoints=[80,160,240],objective=a.mode,seed=10,student_additional_sft=False,validation='GSM8K train7000:7100',test='GSM8K test1200:1300',selection='Best validation accuracy; lower step breaks ties; test not used to choose settings')
    w.state['tokenizer_identity_verified']=True;w.save()
    # Smoke checks actual full-precision master updates, vocabulary paths, and save.
    previous=OUT/('q15_round2_'+a.label+'_'+a.job+'_smoke2_updates.json')
    sm=read(previous)
    assert sm['complete'] and sm['actual_optimizer_steps']==2 and sm['mode']==a.mode
    assert sm['teacher_dtype']=='torch.float16' and sm['student_dtype']=='torch.bfloat16'
    assert sm['updates'][-1]['master_delta_rms']>0
    w.state['reused_successful_smoke']=dict(path=str(previous),sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),note='Prior wrapper failed while enumerating saved checkpoint paths after successful smoke; formal training starts fresh.')
    w.save()
    checkpoints=w.opd('train240',ORIGINAL,240,a.lr,a.port,80,a.mode)
    initial=read(OUT/'initial_val100/gsm8k-results.json')
    candidates=[]
    for step,model in checkpoints.items():
        d,path=w.evaluate('val'+str(step),model)
        comp=compare(initial,d)
        w.state.setdefault('validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        candidates.append((sum(scores(d)),step,str(model)))
        w.save()
    best=max(candidates,key=lambda x:(x[0],-x[1]))
    w.state['selected']=dict(correct=best[0],step=best[1],model=best[2],lr=a.lr,mode=a.mode,gain_pp=float(best[0]-sum(scores(initial))))
    w.finish()
except Exception as error:
    w.fail(error);raise
