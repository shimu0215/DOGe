"""Select clean OPD on validation only; fixed 100-question test is not searched."""
import argparse
import hashlib
import time
from common import *

p=argparse.ArgumentParser()
p.add_argument('--job',required=True)
p.add_argument('--label',required=True)
p.add_argument('--lr',required=True,type=float)
p.add_argument('--port',required=True,type=int)
p.add_argument('--initial',action='store_true')
a=p.parse_args()
w=Worker(a.job,'q15_clean_'+a.label+'_'+a.job)
try:
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(STUDENT)
    r=AutoTokenizer.from_pretrained(ORIGINAL)
    old=AutoTokenizer.from_pretrained('/scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct')
    assert t.get_vocab()==r.get_vocab()==old.get_vocab()
    assert t.chat_template==r.chat_template==old.chat_template
    w.state['protocol']=dict(student=str(STUDENT),teacher=str(ORIGINAL),lr=a.lr,steps=120,checkpoints=[40,80,120],seed=10,student_additional_sft=False,validation='GSM8K train7000:7100',test='GSM8K test1200:1300',selection='Best validation accuracy; lower step breaks ties; test not used to choose settings')
    w.state['tokenizer_identity_verified']=True;w.save()
    if a.initial:
        w.evaluate('initial_val',STUDENT,destination=OUT/'initial_val100')
        w.evaluate('initial_test',STUDENT,'test',1200,100,destination=OUT/'initial_test100')
    # Smoke checks actual full-precision master updates, vocabulary paths, and save.
    w.opd('smoke2',ORIGINAL,2,a.lr,a.port,2)
    checkpoints=w.opd('train120',ORIGINAL,120,a.lr,a.port,40)
    initial=read(OUT/'initial_val100/gsm8k-results.json')
    candidates=[]
    for step,model in checkpoints.items():
        d,path=w.evaluate('val'+str(step),model)
        comp=compare(initial,d)
        w.state.setdefault('validation',{})[str(step)]=dict(**comp,model=str(model),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        candidates.append((sum(scores(d)),step,str(model)))
        w.save()
    best=max(candidates,key=lambda x:(x[0],-x[1]))
    w.state['selected']=dict(correct=best[0],step=best[1],model=best[2],lr=a.lr,gain_pp=float(best[0]-sum(scores(initial))))
    w.finish()
except Exception as error:
    w.fail(error);raise
