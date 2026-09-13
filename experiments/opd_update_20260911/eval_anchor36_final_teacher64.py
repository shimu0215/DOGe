"""Budgeted two fixed-student extensions and optional selected teacher quality, no training."""
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
tag = 'static_anchor36_final_teacher64_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 180
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
    source='static_tail_freq_anti6_anchor36_9916493'
    parent=read(OUT/'static_repair_extension_anchor36_9983837_worker.json');assert parent['complete'] and not parent.get('error')
    state['scope']='Only first64 of expanded400 teacher questions, paired fixed generation; not full400 or proof of noninferiority'
    dest=OUT/(tag+'_teacher64')
    run('teacher64',[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),'--model',str(OUT/source/'model'),'--examples',str(OUT/'static_entropy_slice600_audit_9915409_sft/gsm8k-results.json'),'--output',str(dest),'--limit','64','--modes','greedy','sampling','--max-tokens','512'])
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    score=lambda rows:[int(prediction(x['prediction'].replace(chr(92)+',',' '))[0]==gold(x['ground_truth'])) for x in rows]
    key=lambda rows:[(x['id'],x['prompt'],x['ground_truth']) for x in rows]
    base=OUT/'static_teachers_slice600_9916542_original'
    summary=read(dest/'summary.json');assert summary['complete']
    for mode in ['greedy','sampling']:
        before=[json.loads(x) for x in (base/(mode+'.jsonl')).read_text().splitlines()][:64]
        after=[json.loads(x) for x in (dest/(mode+'.jsonl')).read_text().splitlines()]
        assert len(after)==64 and key(before)==key(after)
        assert read(base/'summary.json')['modes'][mode]['generation']==summary['modes'][mode]['generation']
        x,y=score(before),score(after);state.setdefault('teacher64',{})[mode]=dict(original=sum(x)/64,candidate=sum(y)/64,paired=paired(x,y))
    state.update(complete=True,phase='complete',end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time());raise
finally:
    save()
