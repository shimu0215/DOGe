"""Independent-family native-format CoT SFT and headroom calibration, not token-KL OPD."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();c.OUT=ROOT/'results/generalization_20260913'
w=c.Worker(a.job,'smollm_sft_three_'+a.job,minimum=3600)
try:
 from transformers import AutoTokenizer
 from datasets import load_dataset
 downloaded=c.read(c.OUT/'download_smollm.json');assert downloaded['complete']
 raw=Path(downloaded['models']['smollm2-360m-instruct']['path']);tok=AutoTokenizer.from_pretrained(raw);teacher_tok=AutoTokenizer.from_pretrained(c.ORIGINAL)
 assert tok.get_vocab()!=teacher_tok.get_vocab()
 w.state['scope']='Native-format independent-family student preparation. No teacher training, no cross-tokenizer logit alignment, no OPD or defense result claimed.';w.save()
 original=c.read(c.OUT/'smollm_sft_22480_raw_val100/gsm8k-results.json')
 w.state['reused_raw_evaluation']='Original raw100 complete; prior preparation stopped on two incorrect legacy CoTs, before any SFT.'
 w.state['raw_correct']=sum(c.scores(original));w.save()
 reference=c.read(c.OUT/'coder_transfer_28528_original_teacher_val/gsm8k-results.json')
 assert [(r['id'],r['ground_truth']) for r in original['content']]==[(r['id'],r['ground_truth']) for r in reference['content']]
 w.state['teacher_correct']=sum(c.scores(reference));w.save()
 assert w.state['teacher_correct']>w.state['raw_correct']
 data=Path('/scratch/wzhao20/DOGe-official/data/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/train.jsonl')
 source=[json.loads(s) for s in data.read_text().splitlines()];rejected=[];gsm=load_dataset('openai/gsm8k','main',split='train');prepared_rows=[]
 system='Please reason step by step, and put your final answer within \\boxed{{}}.'
 for row in source:
  n=next(i for i,x in enumerate(row['labels']) if x!=-100)
  assert all(x==-100 for x in row['labels'][:n]) and all(x==y for x,y in zip(row['labels'][n:],row['input_ids'][n:]))
  prompt=teacher_tok.decode(row['input_ids'][:n],skip_special_tokens=False)
  question=prompt.split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
  idx=int(row['id']);assert 0<=idx<1000 and question==gsm[idx]['question']
  response=teacher_tok.decode(row['input_ids'][n:],skip_special_tokens=True)
  predicted=c.prediction(response.replace(chr(92)+',',' '))[0];expected=c.gold(gsm[idx]['answer'])
  if predicted!=expected:
   rejected.append(dict(id=idx,predicted=str(predicted),expected=str(expected),response_tail=response[-500:]));continue
  head=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':question}],tokenize=True,add_generation_prompt=True)
  tail=tok.encode(response,add_special_tokens=False)+[tok.eos_token_id]
  ids=head+tail;assert max(ids)<len(tok)
  prepared_rows.append(dict(id=idx,input_ids=ids,attention_mask=[1]*len(ids),labels=[-100]*len(head)+tail))
 assert len(prepared_rows)>=700 and len(prepared_rows)+len(rejected)==len(source)
 w.state['source_filter']=dict(original_count=len(source),accepted=len(prepared_rows),rejected=rejected,criterion='Existing corrected numeric answer parser, applied before native retokenization. Original historical SFT corpus unchanged.');w.save()
 encoded=c.OUT/(w.tag+'_train.jsonl');encoded.write_text(''.join(json.dumps(r)+'\n' for r in prepared_rows))
 prepared=c.OUT/(w.tag+'_native');prepared.mkdir()
 for f in raw.glob('*.safetensors'):(prepared/f.name).symlink_to(f.resolve())
 for f in raw.glob('*.safetensors.index.json'):(prepared/f.name).symlink_to(f.resolve())
 c.write(prepared/'config.json',c.read(raw/'config.json'));tok.save_pretrained(prepared)
 gen=c.read(raw/'generation_config.json') if (raw/'generation_config.json').exists() else {}
 gen.update(do_sample=False,temperature=1.,top_p=1.,top_k=50,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
 c.write(prepared/'generation_config.json',gen)
 w.state['data']=dict(source=str(data),source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),retokenized=str(encoded),retokenized_sha256=hashlib.sha256(encoded.read_bytes()).hexdigest(),count=len(prepared_rows),native_tokenizer=True,max_tokens=max(len(r['input_ids']) for r in prepared_rows),all_original_ids_in_train_first1000=True,all_cot_answers_correct=True);w.save()
 output=c.OUT/(w.tag+'_training')
 w.run('sft3',[c.PY,'/scratch/wzhao20/DOGe-official/scripts/train_gsm_cot_sft.py','--student-model',str(prepared),'--train-jsonl',str(encoded),'--output-dir',str(output),'--epochs','3','--batch-size','4','--grad-accum','4','--learning-rate','3e-5','--seed','42'])
 summary=c.read(output/'training_summary.json');assert summary['epochs']==3 and summary['global_step']>0
 final=Path(summary['final_dir']);val,vpath=w.evaluate('sft_val100',final)
 w.state['sft']=dict(model=str(final),correct=sum(c.scores(val)),teacher_correct=w.state['teacher_correct'],vs_raw=c.compare(original,val),validation_path=str(vpath));w.state['next']='If useful, design an explicitly cross-tokenizer on-policy objective with its own positive baseline before testing defense. Do not crop logits or claim this SFT check is OPD.';w.finish()
except Exception as e:w.fail(e);raise
