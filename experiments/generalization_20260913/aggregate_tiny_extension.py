"""CPU-only recovery of completed evaluation slices; original failed workers retained."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
R=ROOT/'results/opd_update_20260911';G=ROOT/'results/generalization_20260913';C=ROOT/'results/opd_corrected_20260911'
source='static_entropy4_multifamily_28529'
y=[];xs={'sft':[],'clean':[]};seen=set();evidence=[]
for start,count,job in [(0,200,None),(1000,200,None),(200,200,'28527'),(600,200,'28529'),(800,200,'28530')]:
 candidate=R/(source+'_student_test'+str(start))/'gsm8k-results.json' if job is None else G/('tiny_extension_slice'+str(start)+'_'+job+'_test/gsm8k-results.json')
 doc=c.read(candidate);assert len(doc['content'])==count and doc['generation']['limit']==count
 assert doc['evaluation_split']['indices']==[start,start+count]
 if start in [0,1000]:
  refs={'sft':C/'short_fkl_initial_test0/gsm8k-results.json' if start==0 else ROOT/'results/baseline_20260911/short_initial_test1000/gsm8k-results.json','clean':C/('short_minillm_9871083_selected_test'+str(start))/'gsm8k-results.json'}
 else:
  prefix='static_students_extra200_9871084' if start==200 else 'static_entropy_slice600_audit_9915409'
  refs={label:R/(prefix+'_'+suffix)/'gsm8k-results.json' for label,suffix in [('sft','sft'),('clean','clean_opd')]}
 comparisons={}
 for label,path in refs.items():
  ref=c.read(path);assert ref['generation']['limit']==len(ref['content'])
  # limit is dataset size, not a decoding parameter (audited evaluator lines51-54,117).
  # Greedy batches remain aligned: both slice boundaries are multiples of batch_size8.
  if start>=600 and job is not None:
   assert len(ref['content'])==400 and start%8==0
   ref['content']=ref['content'][start-600:start-600+count]
   ref['generation']['limit']=count
  comparisons[label]=c.compare(ref,doc);xs[label]+=c.scores(ref)
  assert ref['evaluation_split']['evaluator_sha256']==doc['evaluation_split']['evaluator_sha256']
 for row in doc['content']:
  key=(row['prompt'],row['ground_truth']);assert key not in seen;seen.add(key)
 y+=c.scores(doc)
 evidence.append(dict(start=start,count=count,path=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),comparisons=comparisons))
assert len(y)==len(seen)==1000
summary=dict(source=source,n=1000,correct=sum(y),accuracy=sum(y)/1000,reference_accuracy={k:sum(v)/1000 for k,v in xs.items()},comparisons={k:c.paired(v,y) for k,v in xs.items()},evidence=evidence,scores=dict(candidate=y,**xs),training_performed=False,recovery='Original slice600/800 workers failed only because reference limit400 differed from slice limit200. Complete generated results recovered after exact row and decoding checks; workers retained unchanged.',scope='Adaptive exploratory evaluation extension only. Main teacher screen passed but supplementary rawT1 and extra200 greedy failures remain.')
path=G/'tiny_multifamily_student1000_final.json';assert not path.exists();c.write(path,summary)
print(json.dumps({k:summary[k] for k in ['n','correct','accuracy','reference_accuracy','comparisons']},indent=2))
