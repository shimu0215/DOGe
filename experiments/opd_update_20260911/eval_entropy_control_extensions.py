"""Fixed entropy anti0 diagnostic student evaluated on additional slices; no teacher quality relabeling."""
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
tag = 'static_entropy_control_extensions_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 2400
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
    source='static_entropy_control_diagnostic_9916480'
    source_record=read(OUT/(source+'_worker.json'))
    assert source_record['complete'] and source_record['eligible_defense_candidate'] is False
    model=checkpoint(source+'_eval_minillm120_s10')
    state.update(model=str(model),training_performed=False,eligible_defense_candidate=False,
        teacher_results=source_record['teacher_results'],teacher_raw128=source_record['teacher_raw128'],
        protocol='Fixed matched anti0 control student on GSMtest200:400 and600:1000; failed teacher quality remains diagnostic-only')
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    score=lambda d:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in d['content']]
    key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
    for start,count,prefix,entropy_dir in [
        (200,200,'static_students_extra200_9871084','static_entropy_extra200_9915409_entropy'),
        (600,400,'static_entropy_slice600_audit_9915409','static_entropy_slice600_audit_9915409_entropy')]:
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        assert remaining>=count*2, ('Not enough time',start,remaining)
        dest=OUT/(tag+'_test'+str(start))
        run('student_test'+str(start),[os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),
            '--model',str(model),'--output',str(dest),'--split','test','--start',str(start),'--count',str(count)])
        result=read(dest/'gsm8k-results.json')
        assert len(result['content'])==count
        y=score(result)
        comparisons={}
        for label,path in [('clean',OUT/(prefix+'_clean_opd')/'gsm8k-results.json'),
                           ('sft',OUT/(prefix+'_sft')/'gsm8k-results.json'),
                           ('entropy',OUT/entropy_dir/'gsm8k-results.json')]:
            reference=read(path)
            assert reference['generation']==result['generation'] and key(reference)==key(result)
            x=score(reference)
            comparisons[label]=dict(original=sum(x)/len(x),candidate=sum(y)/len(y),paired=paired(x,y))
        state.setdefault('student_results',{})[str(start)]=dict(n=count,accuracy=sum(y)/len(y),comparisons=comparisons)
        save()
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
