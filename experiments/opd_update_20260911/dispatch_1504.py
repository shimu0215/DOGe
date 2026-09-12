"""One-time dispatch after completed whole pipelines; no reservation changes."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/scratch/wzhao20/opd-gate-audit-run-20260909');os.chdir(ROOT)
OUT=ROOT/'results/opd_update_20260911'
finished=['static_self_rl_hard_9916479','oracle_role_reward_top2_9916493','oracle_role_regularizer_top2_9916493','oracle_identity_v2_9870979','static_tail_permutation_coverage_9870979','static_top2_hardrl_9983838','static_tail_anti8_anchor24_9983835','static_tail_last2_anchor24_9983837']
for tag in finished:assert json.loads((OUT/(tag+'_worker.json')).read_text())['complete'],tag
jobs=['9916479','9916493','9870979','9983838','9983835','9983837']
steps=subprocess.check_output(['squeue','--steps','-u','wzhao20','-h','-o','%i'],text=True).splitlines()
for step in steps:
 pieces=step.strip().split('.')
 assert not(len(pieces)==2 and pieces[0] in jobs and pieces[1].isdigit()),step
plans=[('9916479','run_static_top2_head.py',False),('9916493','run_static_top2_gap.py',True),('9916493','run_static_top2_gap_head.py',True),('9870979','eval_tail_last2_anchor24_extensions.py',True),('9870979','eval_tail_anti8_anchor24_extensions.py',True),('9983838','eval_top2_hardrl_extensions.py',False),('9983835','run_static_tail_last2_anti2_anchor24.py',False),('9983837','run_static_tail_permutation_head_anchor24.py',False)]
record=OUT/'dispatch_1504.json';assert not record.exists()
records=[]
for job,script,multi in plans:
 cmd=['srun','--jobid='+job]+(['--exclusive','--exact','--mem=32G'] if multi else ['--overlap','--exact'])+['--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(ROOT/'experiments/opd_update_20260911'/script),'--job',job]
 with (OUT/('dispatch1504_'+script+'_'+job+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 records.append(dict(job=job,script=script,pid=proc.pid,command=cmd,start=time.time()))
 record.write_text(json.dumps(records,indent=2))
 print(job,script,proc.pid,flush=True)
