"""Frozen-teacher transfer to a new 1.5B student; no teacher optimization."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/cross_student_20260913'
OUT.mkdir(exist_ok=True)
STUDENT = OUT / 'qwen2.5-1.5b-instruct'
ORIGINAL = Path('/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct')
DEFENSE = ROOT / 'results/opd_update_20260911/static_tail_freq_anti6_anchor36_9916493/model'
PY = '/scratch/wzhao20/conda_envs/minillm_official/bin/python'
sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction, gold, paired

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def scores(doc):
    return [int(prediction(r['prediction'].replace(chr(92)+',',' '))[0] == gold(r['ground_truth'])) for r in doc['content']]

def compare(before, after):
    key = lambda d: [(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
    assert key(before) == key(after) and before['generation'] == after['generation']
    x,y = scores(before),scores(after)
    return dict(n=len(y),initial_correct=sum(x),candidate_correct=sum(y),initial=sum(x)/len(x),accuracy=sum(y)/len(y),paired=paired(x,y))

class Worker:
    def __init__(self, job, tag, minimum=7200):
        os.chdir(ROOT)
        self.tag = tag
        self.path = OUT / (tag+'_worker.json')
        assert not self.path.exists(), self.path
        self.state = dict(job=job,tag=tag,start=time.time(),complete=False,completed=[],teacher_training_performed=False)
        self.save()
        assert os.environ['SLURM_JOB_ID'] == job
        info = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields = dict(x.split('=',1) for x in info.split() if '=' in x)
        assert fields['JobState'] == 'RUNNING'
        self.deadline = datetime.datetime.fromisoformat(fields['EndTime']).timestamp()
        assert self.deadline-time.time() >= minimum
        import torch
        assert torch.cuda.device_count() == 1
        visible = os.environ['CUDA_VISIBLE_DEVICES']
        devices = [x.split(', ') for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).strip().splitlines()]
        matching = [uuid for index,uuid in devices if index == visible or uuid == visible]
        if len(devices) == 1: matching = [devices[0][1]]
        assert len(matching) == 1
        uuid = matching[0]
        assert not subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        self.state.update(allocation=info,deadline=self.deadline,device=dict(uuid=uuid,visible=visible,step=os.environ['SLURM_STEP_ID']))
        self.save()

    def save(self): write(self.path,self.state)

    def run(self, phase, cmd, env=None):
        assert self.deadline-time.time() >= 120, 'Insufficient allocation time'
        self.state.update(phase=phase,command=cmd);self.save()
        with (OUT/(self.tag+'_'+phase+'.log')).open('x') as log:
            child = subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            self.state['child_pid'] = child.pid;self.save()
            rc = child.wait()
        assert rc == 0, (phase,rc)
        self.state['completed'].append(phase);self.save()

    def evaluate(self, phase, model, split='train', start=7000, count=100, destination=None):
        dest = destination or OUT/(self.tag+'_'+phase)
        self.run(phase,[PY,str(ROOT/'experiments/baseline_20260911/evaluate.py'),'--model',str(model),'--output',str(dest),'--split',split,'--start',str(start),'--count',str(count)])
        d = read(dest/'gsm8k-results.json');assert len(d['content']) == count
        return d, dest/'gsm8k-results.json'

    def opd(self, phase, teacher, steps, lr, port, interval):
        label = self.tag+'_'+phase
        env = os.environ.copy()
        for k in list(env):
            if k == 'PROXY' or k.startswith('INTERNAL_'): del env[k]
        env.update(INTERNAL_TEACHER=str(teacher),INTERNAL_STUDENT=str(STUDENT),INTERNAL_LABEL=label,
                   INTERNAL_SEED='10',INTERNAL_PPO_SEED='42',INTERNAL_LM_SEED='7',INTERNAL_PORT=str(port),
                   BASELINE_STEPS=str(steps),BASELINE_LR=str(lr),CORRECTED_MODE='minillm',
                   CORRECTED_SAVE_INTERVAL=str(interval),CORRECTED_RECORD=str(OUT/(label+'_updates.json')),
                   CORRECTED_STUDENT_VOCAB=str(read(STUDENT/'config.json')['vocab_size']))
        self.run(phase,['bash',str(ROOT/'experiments/opd_corrected_20260911/opd.sh')],env)
        report = read(OUT/(label+'_updates.json'))
        assert report['complete'] and report['actual_optimizer_steps'] == steps
        assert report['teacher_dtype'] == 'torch.float16' and report['student_dtype'] == 'torch.bfloat16'
        assert report['updates'][-1]['master_delta_rms'] > 0
        paths = {}
        for step in range(interval,steps+1,interval):
            found = list((ROOT/'results/opd_corrected_20260911'/(label+'_opd')).glob('**/'+str(step)+'/pytorch_model.bin'))
            assert len(found) == 1, (step,found)
            paths[step] = found[0].parent
        return paths

    def finish(self):
        self.state.update(complete=True,phase='complete',end=time.time());self.save()

    def fail(self,error):
        self.state.update(error=repr(error),end=time.time());self.save()
