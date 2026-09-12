"""Fixed entropy teacher scoring audit and matched students on GSM8K test600:1000."""
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
tag = 'static_entropy_slice600_audit_' + a.job
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
    models = [
        ('entropy', checkpoint('static_entropy_increase_9916494_eval_minillm120_s10')),
        ('sft', ROOT/'results/baseline_20260911/short_sft/checkpoint-49'),
        ('clean_opd', checkpoint('short_minillm_9871083_clean240_s10_opd')),
    ]
    for label,model in models:
        assert (model/'config.json').exists(),model
    state.update(models={k:str(v) for k,v in models}, training_performed=False,
        protocol='Three fixed student checkpoints on GSM8K test600:1000, greedy512 BF16, no student/teacher training',
        limitation='Another fixed GSM8K slice, same model/dataset; exploratory evaluation, not cross-model or cross-dataset evidence')
    sys.path.insert(0,str(ROOT/'experiments/gate_audit_20260909'))
    from corrected_numeric_audit import prediction,gold,paired
    source = OUT/'static_entropy_increase_9916494'
    manifest = read(source/'manifest.json')
    assert manifest['complete'] and manifest['plain_export_verified']
    assert manifest['student_model_loaded'] is False and manifest['student_parameter_signal'] is False
    assert read(OUT/'static_entropy_increase_9916494_worker.json')['complete']
    diagnostic = OUT/(tag+'_scoring64')
    run('heldout_scoring64', [os.environ['PY'],str(Path(__file__).with_name('audit_teacher_scoring_v2.py')),
        '--teacher',os.environ['TEACHER'],'--candidates',str(source/'model'),
        '--context',str(OUT/'static_mixctx_top2_9870980_context'),'--output',str(diagnostic),'--examples','64'])
    assert read(diagnostic/'manifest.json')['complete']
    state['scoring_manifest']=str(diagnostic/'manifest.json')
    state['teacher_generation_scope']='No new teacher generation evaluation on test600:1000 in this pipeline; previous teacher main/extra quality is reported separately.'
    save()
    outputs={}; scores={}
    for label,model in models:
        remaining=datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-time.time()
        assert remaining>=720, ('Not enough time for another400question evaluation',label,remaining)
        dest=OUT/(tag+'_'+label)
        run(label, [os.environ['PY'],str(ROOT/'experiments/baseline_20260911/evaluate.py'),
            '--model',str(model),'--output',str(dest),'--split','test','--start','600','--count','400'])
        result=read(dest/'gsm8k-results.json')
        assert len(result['content'])==400
        if outputs:
            ref=next(iter(outputs.values()))
            assert result['generation']==ref['generation']
            key=lambda d:[(r['id'],r['prompt'],r['ground_truth']) for r in d['content']]
            assert key(result)==key(ref)
        outputs[label]=result
        scores[label]=[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in result['content']]
        state.setdefault('scores',{})[label]=dict(n=400,correct=sum(scores[label]),accuracy=sum(scores[label])/400)
        save()
    state['comparisons']={label:dict(vs_clean=paired(scores['clean_opd'],scores[label]),
        vs_sft=paired(scores['sft'],scores[label])) for label,_ in models}
    save()
    state.update(complete=True, phase='complete', end=time.time())
except Exception as error:
    state.update(error=repr(error), end=time.time())
    raise
finally:
    save()
