"""Fixed anti0 diagnostic teacher on GSM8K test600:1000; no training."""
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
tag = 'static_control_teacher_slice600_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 1500
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
    source='static_entropy_control_9916480'
    source_state=read(OUT/(source+'_worker.json'))
    manifest=read(OUT/source/'manifest.json')
    assert source_state['complete'] and not source_state['teacher_point_tolerance_pass']
    assert manifest['complete'] and manifest['plain_export_verified']
    assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
    examples=OUT/'static_entropy_slice600_audit_9915409_sft/gsm8k-results.json'
    assert len(read(examples)['content'])==400
    state.update(training_performed=False,eligible_defense_candidate=False,
        source_quality_pass=False,protocol='Fixed anti0 diagnostic teacher, GSMtest600:1000, FP16 greedy and standard sampling512; no quality relabeling')
    dest=OUT/tag
    run('teacher400',[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
        '--model',str(OUT/source/'model'),'--examples',str(examples),'--output',str(dest),
        '--limit','400','--modes','greedy','sampling','--max-tokens','512'])
    summary=read(dest/'summary.json')
    assert summary['complete']
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    original=OUT/'static_teachers_slice600_9916542_original'
    original_complete=(original/'summary.json').exists() and read(original/'summary.json').get('complete',False)
    for mode in ['greedy','sampling']:
        rows=[json.loads(line) for line in (dest/(mode+'.jsonl')).read_text().splitlines()]
        key=lambda rs:[(r['id'],r['prompt'],r['ground_truth']) for r in rs]
        assert len(rows)==400 and key(rows)==key(read(examples)['content'])
        score=lambda rs:[int(prediction(r['prediction'].replace(chr(92)+',',' '))[0]==gold(r['ground_truth'])) for r in rs]
        y=score(rows)
        result=dict(n=400,accuracy=sum(y)/400)
        if original_complete:
            baseline=[json.loads(line) for line in (original/(mode+'.jsonl')).read_text().splitlines()]
            assert key(baseline)==key(rows)
            assert summary['modes'][mode]['generation']==read(original/'summary.json')['modes'][mode]['generation']
            x=score(baseline)
            result.update(original=sum(x)/400,paired=paired(x,y))
        state.setdefault('teacher_results',{})[mode]=result
    state.update(complete=True,phase='complete',end=time.time(),original_comparison_complete=original_complete)
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
