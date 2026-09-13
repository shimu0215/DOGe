"""Budgeted repaired-teacher fixed-student extension, no training."""
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
p.add_argument('--teacher-source', required=True)
p.add_argument('--student-source', required=True)
p.add_argument('--label', required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/opd_update_20260911'
CORRECTED = ROOT / 'results/opd_corrected_20260911'
tag = 'static_repair_extension_' + a.label + '_' + a.job
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
    assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time() >= 520
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
    score=lambda rows:[int(prediction(x['prediction'].replace(chr(92)+',',' '))[0]==gold(x['ground_truth'])) for x in rows]
    key=lambda rows:[(x['id'],x['prompt'],x['ground_truth']) for x in rows]
    candidates=[(a.label,a.student_source)]
    state.update(training_performed=False,protocol='Budgeted fixed independent student extra200 and extra400, unchanged evaluation. No teacher evaluation in this worker; skipped phases explicit, adaptive exploratory followup.',sources=candidates)
    summaries={}
    for label,source in candidates:
        worker=read(OUT/(source+'_worker.json'));assert worker['complete'] and worker['teacher_point_tolerance_pass'] and not worker.get('error')
        m=read(OUT/a.teacher_source/'manifest.json');assert m['complete'] and m['plain_export_verified'] and m['teacher']==os.environ['TEACHER']
        assert not any(m[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
        model=checkpoint(source+'_eval_minillm120_s10')
        extra={}
        for start,count,prefix in [(200,200,'static_students_extra200_9871084'),(600,400,'static_entropy_slice600_audit_9915409')]:
            remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
            if remaining<count*2+120:
                state.setdefault('skipped_budget',[]).append(dict(candidate=label,start=start,remaining=remaining));save();continue
            dest=OUT/(tag+'_'+label+'_student_test'+str(start))
            run(label+'_student_test'+str(start),[os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),'--model',str(model),'--output',str(dest),'--split','test','--start',str(start),'--count',str(count)])
            doc=read(dest/'gsm8k-results.json');assert len(doc['content'])==count;extra[start]=dest/'gsm8k-results.json'
            y=score(doc['content']);comparison={}
            for ref in ['sft','clean_opd']:
                other=read(OUT/(prefix+'_'+ref)/'gsm8k-results.json');assert key(other['content'])==key(doc['content']) and other['generation']==doc['generation']
                comparison[ref]=paired(score(other['content']),y)
            state.setdefault('student_results',{}).setdefault(label,{})[str(start)]=dict(n=count,accuracy=sum(y)/count,comparisons=comparison);save()
        if len(extra)!=2:continue
        paths=[OUT/(source+'_student_test'+str(i))/'gsm8k-results.json' for i in [0,1000]]+[extra[200],extra[600]]
        refs={
            'sft':[CORRECTED/'short_fkl_initial_test0/gsm8k-results.json',ROOT/'results/baseline_20260911/short_initial_test1000/gsm8k-results.json',OUT/'static_students_extra200_9871084_sft/gsm8k-results.json',OUT/'static_entropy_slice600_audit_9915409_sft/gsm8k-results.json'],
            'clean':[CORRECTED/('short_minillm_9871083_selected_test'+str(i))/'gsm8k-results.json' for i in [0,1000]]+[OUT/'static_students_extra200_9871084_clean_opd/gsm8k-results.json',OUT/'static_entropy_slice600_audit_9915409_clean_opd/gsm8k-results.json']}
        ys=[];xs={k:[] for k in refs};unique=set();hashes=[]
        for j,path in enumerate(paths):
            doc=read(path);assert len(doc['content'])==[200,200,200,400][j]
            for row in doc['content']:
                q=(row['prompt'],row['ground_truth']);assert q not in unique;unique.add(q)
            ys+=score(doc['content']);hashes.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for ref,ps in refs.items():
                other=read(ps[j]);assert key(other['content'])==key(doc['content']) and other['generation']==doc['generation'];xs[ref]+=score(other['content'])
        assert len(unique)==len(ys)==1000
        summaries[label]=dict(source=source,n=1000,correct=sum(ys),accuracy=sum(ys)/1000,comparisons={ref:paired(v,ys) for ref,v in xs.items()},paths=hashes)
        state['student1000']=summaries;save()
    state['teacher_evaluation_performed']=False
    state.update(complete=True,phase='complete',end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time());raise
finally:
    save()
