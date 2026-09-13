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
tag = 'repair_completion_' + a.job
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
    teachers={'anchor36':'static_tail_freq_anti6_anchor36_9916493','answer4':'static_tail_freq_anti6_answer4_9916493'}
    for label,source in teachers.items():
        manifest=read(OUT/source/'manifest.json');assert manifest['complete'] and manifest['plain_export_verified'] and manifest['teacher']==os.environ['TEACHER']
        assert not any(manifest[k] for k in ['student_model_loaded','student_parameter_signal','student_outcome_reward'])
    state.update(training_performed=False,protocol='Complete answer4 missing student400 to aggregate1000, then BOTH fixed repaired teachers expanded400 greedy/sampling. Same protocols, no checkpoint selection or teacher updates.')
    source=teachers['answer4']+'_standard_eval_9983838'
    previous=read(OUT/'static_repair_extension_answer4_9983838_worker.json');assert previous['complete'] and previous.get('skipped_budget') and not previous.get('error')
    model=checkpoint(source+'_eval_minillm120_s10')
    dest=OUT/(tag+'_answer4_student_test600')
    run('answer4_student_test600',[os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),'--model',str(model),'--output',str(dest),'--split','test','--start','600','--count','400'])
    paths=[OUT/(source+'_student_test'+str(i))/'gsm8k-results.json' for i in [0,1000]]+[OUT/'static_repair_extension_answer4_9983838_answer4_student_test200/gsm8k-results.json',dest/'gsm8k-results.json']
    refs={
        'sft':[CORRECTED/'short_fkl_initial_test0/gsm8k-results.json',ROOT/'results/baseline_20260911/short_initial_test1000/gsm8k-results.json',OUT/'static_students_extra200_9871084_sft/gsm8k-results.json',OUT/'static_entropy_slice600_audit_9915409_sft/gsm8k-results.json'],
        'clean':[CORRECTED/('short_minillm_9871083_selected_test'+str(i))/'gsm8k-results.json' for i in [0,1000]]+[OUT/'static_students_extra200_9871084_clean_opd/gsm8k-results.json',OUT/'static_entropy_slice600_audit_9915409_clean_opd/gsm8k-results.json']}
    y=[];xs={k:[] for k in refs};unique=set();hashes=[]
    for j,p in enumerate(paths):
        d=read(p);assert len(d['content'])==[200,200,200,400][j]
        for r in d['content']:
            question=(r['prompt'],r['ground_truth']);assert question not in unique;unique.add(question)
        y+=score(d['content']);hashes.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        for label,ps in refs.items():
            b=read(ps[j]);assert key(b['content'])==key(d['content']) and b['generation']==d['generation'];xs[label]+=score(b['content'])
    assert len(y)==len(unique)==1000
    state['answer4_student1000']=dict(n=1000,correct=sum(y),accuracy=sum(y)/1000,comparisons={k:paired(x,y) for k,x in xs.items()},paths=hashes);save()
    for label,source in teachers.items():
        assert datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()>=1800
        dest=OUT/(tag+'_'+label+'_teacher400')
        run(label+'_teacher400',[os.environ['PY'],str(ROOT/'experiments/internalize_20260910/evaluate_plain.py'),'--model',str(OUT/source/'model'),'--examples',str(OUT/'static_entropy_slice600_audit_9915409_sft/gsm8k-results.json'),'--output',str(dest),'--limit','400','--modes','greedy','sampling','--max-tokens','512'])
        summary=read(dest/'summary.json');assert summary['complete']
        baseline=OUT/'static_teachers_slice600_9916542_original'
        for mode in ['greedy','sampling']:
            before=[json.loads(x) for x in (baseline/(mode+'.jsonl')).read_text().splitlines()];after=[json.loads(x) for x in (dest/(mode+'.jsonl')).read_text().splitlines()]
            assert len(after)==400 and key(before)==key(after)
            assert read(baseline/'summary.json')['modes'][mode]['generation']==summary['modes'][mode]['generation']
            x,y=score(before),score(after);state.setdefault('teacher400',{}).setdefault(label,{})[mode]=dict(n=400,original=sum(x)/400,candidate=sum(y)/400,paired=paired(x,y))
        state.setdefault('teacher400_point_pass',{})[label]=state['teacher400'][label]['greedy']['paired']['delta_pp']>=0 and state['teacher400'][label]['sampling']['paired']['delta_pp']>=-1
        save()
    state.update(complete=True,phase='complete',end=time.time())
except Exception as error:
    state.update(error=repr(error),end=time.time());raise
finally:
    save()
