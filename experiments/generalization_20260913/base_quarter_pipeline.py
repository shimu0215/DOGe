"""Unseen base checkpoint: native diagnostic, modest CoT SFT, clean OPD, fixed defense."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
from matched_worker import Worker
c.Worker=Worker
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
c.OUT=ROOT/'results/generalization_20260913';c.OUT.mkdir(exist_ok=True)
w=c.Worker(a.job,'base_quarter_'+a.job,minimum=5400)
try:
 download=c.read(c.OUT/'download.json');assert download['complete']
 entry=download['models']['qwen2.5-0.5b-base'];assert entry['exact_teacher_vocab_mapping']
 raw=Path(entry['path'])
 from transformers import AutoTokenizer
 base_tok=AutoTokenizer.from_pretrained(raw);teacher_tok=AutoTokenizer.from_pretrained(c.ORIGINAL)
 assert base_tok.get_vocab()==teacher_tok.get_vocab()
 w.state['scope']='Unseen pretrained checkpoint, same Qwen family; NOT a cross-family proof. Teacher remains fixed and has no new negative examples.'
 w.state['download']=entry;w.save()
 w.run('raw_native_val',[c.PY,str(Path(__file__).with_name('eval_base_native.py')),'--model',str(raw),'--output',str(c.OUT/(w.tag+'_raw_native_val'))])
 raw_result=c.read(c.OUT/(w.tag+'_raw_native_val/results.json'))
 teacher_val,_=w.evaluate('original_teacher_val',c.ORIGINAL)
 w.state['raw_gap']=dict(raw_correct=raw_result['correct'],teacher_correct=sum(c.scores(teacher_val)),n=100,format_note='raw base completion versus teacher native chat; same questions and token budget')
 w.save()
 if sum(c.scores(teacher_val))<=raw_result['correct']:
  w.state['stopped']='Teacher does not outperform raw student in this diagnostic; revise dataset or student before OPD.';w.finish();sys.exit(0)
 # Prepare a separate format-compatible checkpoint; never alter downloaded weights.
 prepared=c.OUT/(w.tag+'_base_chat_format');prepared.mkdir()
 for source in raw.glob('*.safetensors'):(prepared/source.name).symlink_to(source.resolve())
 for source in raw.glob('*.safetensors.index.json'):(prepared/source.name).symlink_to(source.resolve())
 conf=c.read(raw/'config.json')
 conf.update(eos_token_id=teacher_tok.eos_token_id,pad_token_id=teacher_tok.pad_token_id)
 c.write(prepared/'config.json',conf)
 teacher_tok.save_pretrained(prepared)
 gen=c.read(raw/'generation_config.json') if (raw/'generation_config.json').exists() else {}
 gen.update(eos_token_id=teacher_tok.eos_token_id,pad_token_id=teacher_tok.pad_token_id,do_sample=False)
 c.write(prepared/'generation_config.json',gen)
 data=Path('/scratch/wzhao20/DOGe-official/data/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/train.jsonl')
 token_rows=[json.loads(line) for line in data.read_text().splitlines()]
 assert all(max(row['input_ids'])<conf['vocab_size'] for row in token_rows)
 w.state['sft_protocol']=dict(raw_weight_model=str(raw),prepared_format=str(prepared),data=str(data),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),epochs=0.25,learning_rate=2e-5,selection='Checkpoint nearest 40% on validation; ties earlier epoch. No test selection.',format_change='Teacher ChatML tokenizer and stop configuration, identical token-to-ID mapping, no weight changes before SFT.')
 w.save()
 sft=c.OUT/(w.tag+'_sft')
 w.run('sft2',[c.PY,'/scratch/wzhao20/DOGe-official/scripts/train_gsm_cot_sft.py','--student-model',str(prepared),'--train-jsonl',str(data),'--output-dir',str(sft),'--epochs','0.25','--batch-size','4','--grad-accum','4','--learning-rate','2e-5','--seed','42'])
 assert c.read(sft/'training_summary.json')['epochs']==0.25
 candidates=[]
 for path in sorted(sft.glob('checkpoint-*')):
  val,vpath=w.evaluate('sft_'+path.name+'_val',path)
  epoch=c.read(path/'trainer_state.json')['epoch'];correct=sum(c.scores(val));candidates.append((abs(correct-40),epoch,path,val))
  w.state.setdefault('sft_validation',{})[path.name]=dict(correct=correct,epoch=epoch,path=str(vpath));w.save()
 assert len(candidates)==1
 _,epoch,student,initial=min(candidates,key=lambda x:(x[0],x[1]))
 c.STUDENT=student
 w.state['selected_sft']=dict(model=str(student),epoch=epoch,correct=sum(c.scores(initial)));w.save()
 if sum(c.scores(teacher_val))<=sum(c.scores(initial)):
  w.state['stopped']='No teacher headroom over selected SFT student';w.finish();sys.exit(0)
 checkpoints=w.opd('clean120',c.ORIGINAL,120,1e-6,31341,40,'minillm')
 choices=[]
 for step,model in checkpoints.items():
  val,vpath=w.evaluate('clean_val'+str(step),model);comp=c.compare(initial,val)
  w.state.setdefault('clean_validation',{})[str(step)]=dict(**comp,model=str(model),path=str(vpath));w.save()
  choices.append((comp['candidate_correct'],step,model))
 correct,step,model=max(choices,key=lambda x:(x[0],-x[1]))
 w.state['selected_opd']=dict(correct=correct,step=step,model=str(model),gain_pp=correct-sum(c.scores(initial)),lr=1e-6,mode='minillm');w.save()
 if correct-sum(c.scores(initial))<3:
  w.state['stopped']='Clean validation gain below 3pp, no defense inference; continue baseline research';w.finish();sys.exit(0)
 initial_test,_=w.evaluate('initial_test100',student,'test',1200,100)
 clean_test,_=w.evaluate('clean_test100',model,'test',1200,100)
 w.state['clean_test']=c.compare(initial_test,clean_test);w.save()
 if w.state['clean_test']['paired']['delta_pp']<=0:
  w.state['stopped']='Selected clean test has no positive gain; no test-driven reselection or defense claim';w.finish();sys.exit(0)
 manifest=c.read(c.DEFENSE.parent/'manifest.json')
 assert manifest['complete'] and manifest['plain_export_verified'] and manifest['teacher']==str(c.ORIGINAL)
 assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
 context=c.read(ROOT/'results/opd_update_20260911/static_mixctx_top2_9870980_context/manifest.json')
 assert isinstance(context['student'],list) and all('instruct' in s.lower() or 'sft' in s.lower() for s in context['student'])
 w.state['defense_teacher']=dict(model=str(c.DEFENSE),manifest_sha256=hashlib.sha256((c.DEFENSE.parent/'manifest.json').read_bytes()).hexdigest(),negative_generators=context['student'],new_student_data_used=False);w.save()
 defensive=w.opd('defense'+str(step),c.DEFENSE,step,1e-6,31343,step,'minillm')[step]
 final,path=w.evaluate('defense_test100',defensive,'test',1200,100)
 w.state['defense_test']=dict(vs_initial=c.compare(initial_test,final),vs_clean=c.compare(clean_test,final),path=str(path));w.finish()
except Exception as e:
 w.fail(e);raise
