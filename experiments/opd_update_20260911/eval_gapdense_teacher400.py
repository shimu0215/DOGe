"""Fixed freqrl student extra600 and teacher new400; no training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--job', required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = 'gapdense_teacher400_' + a.job
record = OUT / (tag + '_worker.json')
assert not record.exists(), record
state = dict(start=time.time(), job=a.job, complete=False, completed=[],
             method_scope='No student model or parameter-derived signal in teacher training; fixed negative CoTs only')

def read(path):
    return json.loads(path.read_text())

def save():
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(record)

def run(phase, cmd, env=None):
    state.update(phase=phase, command=cmd)
    save()
    with (OUT / (tag + '_' + phase + '.log')).open('x') as log:
        child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        save()
        code = child.wait()
    if code:
        raise RuntimeError(phase + ' exit=' + str(code))
    state['completed'].append(phase)
    save()

try:
    save()
    assert os.environ['SLURM_JOB_ID'] == a.job
    info = subprocess.check_output(['scontrol', 'show', 'job', a.job, '-o'], text=True)
    fields = dict(x.split('=', 1) for x in info.split() if '=' in x)
    assert fields['JobState'] == 'RUNNING'
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 7200
    import torch
    assert torch.cuda.device_count() == 1
    devices = [line.strip().split(', ') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).strip().splitlines()]
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    assert ',' not in visible
    if len(devices) == 1:
        device_uuid = devices[0][1]
    else:
        matched = [uuid for index, uuid in devices if index == visible or uuid == visible]
        assert len(matched) == 1, (visible, devices)
        device_uuid = matched[0]
    assert not subprocess.check_output(['nvidia-smi', '-i', device_uuid,
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    state['allocation'] = info
    state['device'] = dict(visible=os.environ['CUDA_VISIBLE_DEVICES'], step=os.environ['SLURM_STEP_ID'],
                          uuid=device_uuid)
    save()
    def checkpoint(prefix):
        found = list(CORRECTED.glob(prefix+'*/**/120/pytorch_model.bin'))
        assert len(found) == 1, (prefix,found)
        return found[0].parent
    import hashlib
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    score=lambda rows:[int(prediction(r['prediction'].replace(chr(92)+',',' '))[0]==gold(r['ground_truth'])) for r in rows]
    key=lambda rows:[(r['id'],r['prompt'],r['ground_truth']) for r in rows]
    teachers={'gapdense':'static_tail_round2_gapdense_28527'}
    for label,source in teachers.items():
        manifest=read(OUT/source/'manifest.json');assert manifest['complete'] and manifest['plain_export_verified'] and manifest['teacher']==os.environ['TEACHER']
        assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
    state.update(training_performed=False,protocol='Both fixed teachers raw temperature1 top_p1 top_k0 rep1, same400 as original reference; no training or selection.')
    examples=OUT/'static_entropy_slice600_audit_9915409_sft/gsm8k-results.json'
    baseline=OUT/'static_teachers_raw600_9916542_original/raw.jsonl'
    assert examples.exists()
    if not baseline.exists():
        # Resolve the recorded original raw400 reference explicitly; never silently substitute another slice.
        candidates=list(OUT.glob('*original/raw.jsonl'))
        matches=[]
        for candidate in candidates:
            rows=[json.loads(x) for x in candidate.read_text().splitlines()]
            if len(rows)==400 and key(rows)==key(read(examples)['content']):matches.append(candidate)
        assert len(matches)==1,(str(baseline),matches)
        baseline=matches[0]
    before=[json.loads(x) for x in baseline.read_text().splitlines()]
    state['original_reference']=dict(path=str(baseline),sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    for label,source in teachers.items():
        dest=OUT/(tag+'_'+label+'_raw400')
        run(label+'_raw400',[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),'--model',str(OUT/source/'model'),'--examples',str(examples),'--output',str(dest),'--limit','400','--modes','raw','--max-tokens','512'])
        assert read(dest/'summary.json')['complete']
        path=dest/'raw.jsonl';after=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(after)==400 and key(before)==key(after)
        x,y=score(before),score(after)
        state.setdefault('raw400',{})[label]=dict(n=400,original_correct=sum(x),candidate_correct=sum(y),paired=paired(x,y),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        standard=OUT/(tag+'_'+label+'_standard400')
        run(label+'_standard400',[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),'--model',str(OUT/source/'model'),'--examples',str(examples),'--output',str(standard),'--limit','400','--modes','greedy','sampling','--max-tokens','512'])
        for mode in ['greedy','sampling']:
            ref=OUT/'static_teachers_slice600_9916542_original'
            assert read(ref/'summary.json')['modes'][mode]['generation']==read(standard/'summary.json')['modes'][mode]['generation']
            b=[json.loads(s) for s in (ref/(mode+'.jsonl')).read_text().splitlines()];a=[json.loads(s) for s in (standard/(mode+'.jsonl')).read_text().splitlines()]
            assert len(a)==400 and key(b)==key(a)
            x,y=score(b),score(a)
            state.setdefault('standard400',{}).setdefault(label,{})[mode]=dict(n=400,original_correct=sum(x),candidate_correct=sum(y),paired=paired(x,y))
        save()
    state.update(complete=True,phase='complete',end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time());raise
finally:
    save()
