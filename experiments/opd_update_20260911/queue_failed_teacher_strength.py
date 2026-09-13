"""Wait for an existing whole quality pipeline, then run one external diagnostic.

No periodic GPU inventory: waits on its known login-node srun PID.
"""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--source',required=True);p.add_argument('--wait-pid',required=True,type=int);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);r=ROOT/'results/opd_update_20260911';record=r/('post_quality_strength_'+a.job+'.json');assert not record.exists();state=dict(start=time.time(),job=a.job,source=a.source,prior_srun_pid=a.wait_pid,complete=False,phase='waiting_whole_pipeline',scope='Automatic whole-pipeline completion handoff; external strength diagnostic cannot pass failed teacher quality')
def save():
 tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
try:
 save();source=r/(a.source+'_worker.json');w=json.loads(source.read_text())
 if not w['complete'] and not w.get('error'):
  proc=Path('/proc')/str(a.wait_pid)/'cmdline'
  if proc.exists():
   cmd=proc.read_bytes().replace(b'\x00',b' ').decode();assert 'srun' in cmd and ('--jobid='+a.job) in cmd and 'run_static_entropy2_' in cmd
   subprocess.run(['tail','--pid='+str(a.wait_pid),'-f','/dev/null'],check=True)
 w=json.loads(source.read_text());assert w['complete'] and not w.get('error')
 if w.get('teacher_point_tolerance_pass') is not False:
  state.update(phase='skipped',reason='Source did not finish with a teacher-quality failure; do not duplicate its conditional student pipeline',complete=True);save();raise SystemExit(0)
 info=subprocess.check_output(['scontrol','show','job',a.job,'-o'],text=True);fields=dict(x.split('=',1) for x in info.split() if '=' in x)
 if fields['JobState']!='RUNNING' or datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()<1800:
  state.update(phase='skipped',reason='Allocation not running or insufficient1800second budget',complete=True);save();raise SystemExit(0)
 steps=subprocess.check_output(['squeue','--steps','-j',a.job,'-h','-o','%i'],text=True).split();assert not any(x.split('.')[-1].isdigit() for x in steps),steps
 cmd=['srun','--jobid='+a.job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__).with_name('eval_failed_teacher_strength.py')),'--job',a.job,'--source',a.source]
 with (r/('post_quality_strength_'+a.job+'_srun.log')).open('x') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);state.update(phase='external_strength',srun_pid=proc.pid,command=cmd);save();rc=proc.wait()
 state.update(phase='complete' if rc==0 else 'failed',complete=True,returncode=rc,end=time.time());save()
except Exception as e:
 state.update(phase='failed',error=repr(e),end=time.time());save();raise
