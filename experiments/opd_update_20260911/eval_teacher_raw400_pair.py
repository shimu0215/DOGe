"""Frozen original and frequentRL teachers: raw400 paired generation, no training."""
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
tag = 'teacher_raw400_pair_' + a.job
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
    source='static_tail_last2_frequent_rl_9983837'
    source_record=read(OUT/(source+'_worker.json'))
    manifest=read(OUT/source/'manifest.json')
    assert source_record['complete'] and source_record['teacher_point_tolerance_pass']
    assert manifest['complete'] and manifest['plain_export_verified']
    assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
    model=checkpoint(source+'_eval_minillm120_s10')
    state.update(training_performed=False,protocol='Frozen originalteacher and frequentRL raw T1,p1,k0,rep1 on400 extra questions; no training, supplementary robustness only.',
        limitation='Adaptive same-dataset/model extension. Main and prior extra teacher quality does not establish new400 preservation. Retain all results.')
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    score=lambda rows:[int(prediction(r['prediction'].replace(chr(92)+',',' '))[0]==gold(r['ground_truth'])) for r in rows]
    key=lambda rows:[(r['id'],r['prompt'],r['ground_truth']) for r in rows]
    outputs={}
    for label,teacher in [('original',Path(os.environ['TEACHER'])),('frequent_rl',OUT/source/'model')]:
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        assert remaining>=1200, ('Insufficient raw400 budget',label,remaining)
        dest=OUT/(tag+'_'+label)
        run('raw400_'+label,[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),
            '--model',str(teacher),'--examples',str(OUT/'static_entropy_slice600_audit_9915409_sft/gsm8k-results.json'),
            '--output',str(dest),'--limit','400','--modes','raw','--max-tokens','512'])
        summary=read(dest/'summary.json');assert summary['complete']
        rows=[json.loads(x) for x in (dest/'raw.jsonl').read_text().splitlines()];assert len(rows)==400
        outputs[label]=(rows,summary['modes']['raw']['generation'])
    before,after=outputs['original'][0],outputs['frequent_rl'][0]
    assert key(before)==key(after) and outputs['original'][1]==outputs['frequent_rl'][1]
    assert len(set((x['prompt'],x['ground_truth']) for x in before))==400
    x,y=score(before),score(after)
    state['raw400']=dict(original=sum(x)/400,candidate=sum(y)/400,paired=paired(x,y),generation=outputs['original'][1])
    state.update(complete=True,phase='complete',end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time())
    raise
finally:
    save()
