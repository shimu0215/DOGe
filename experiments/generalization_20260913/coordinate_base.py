"""Claim the third generalization GPU only after its complete prior pipeline ends."""
import datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'results/generalization_20260913';record=OUT/'coordinator_base.json';assert not record.exists()
x=dict(start=time.time(),complete=False,phase='waiting_download_and_raw400_parent',job='23369')
def save():
 tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(record)
save()
try:
 while time.time()-x['start']<10800:
  parent_path=ROOT/'results/opd_update_20260911/repair_raw400_23369_worker.json'
  parent=json.loads(parent_path.read_text());download=json.loads((OUT/'download.json').read_text())
  if parent.get('error') or download.get('error'):raise RuntimeError({'parent':parent.get('error'),'download':download.get('error')})
  if not parent.get('complete') or not download.get('complete'):time.sleep(30);continue
  job=x['job'];steps=subprocess.check_output(['squeue','--steps','-j',job,'-h','-o','%i'],text=True).split()
  if any(s.rsplit('.',1)[-1].isdigit() for s in steps):time.sleep(10);continue
  info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True);fields=dict(s.split('=',1) for s in info.split() if '=' in s)
  if fields.get('JobState')!='RUNNING':raise RuntimeError('Destination no longer running')
  remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
  if remaining<12660:raise RuntimeError('Insufficient remaining budget for complete base pipeline')
  command=['srun','--jobid='+job,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','--gres=gpu:1',os.environ['PY'],str(Path(__file__).with_name('base_pipeline.py')),'--job',job]
  with (OUT/'base_transfer_23369_srun.log').open('x') as log:
   child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  x.update(phase='base_pipeline_dispatched',pid=child.pid,command=command,remaining=remaining,parent_complete=str(parent_path));break
 else:raise TimeoutError('Dependency wait budget exhausted')
 x.update(complete=True,end=time.time());save()
except Exception as e:
 x.update(error=repr(e),end=time.time());save();raise
