"""Fixed seed 11/12 replications after the primary queue, one allocated GPU per arm."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = ROOT / 'results/learning_loop_20260910'
sys.path.insert(0, str(ROOT / 'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction, gold, paired

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=['directional', 'preservation_control', 'strong_rank'], required=True)
a = p.parse_args()
job, node, end, port = {
    'directional': ('9801342', 'gpu010', '2026-09-11T05:11:13-04:00', '30103'),
    'preservation_control': ('9795227', 'gpu008', '2026-09-11T02:03:00-04:00', '30101'),
    'strong_rank': ('9801341', 'gpu019', '2026-09-11T05:42:19-04:00', '30105'),
}[a.arm]
deadline = datetime.datetime.fromisoformat(end).timestamp()
PY = os.environ['PY']
os.environ['INTERNAL_PORT'] = port
record = OUT / (a.arm + '_repeat_queue.json')
# Exclusive creation prevents duplicate drivers even if launched together.
with record.open('x') as f:
    json.dump({'phase': 'initializing'}, f)
state = dict(start=time.time(), driver_pid=os.getpid(), arm=a.arm, job=job, node=node,
             deadline=deadline, seeds=[11, 12], completed_phases=[], results={})

def save():
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(record)

def read(path):
    return json.loads(path.read_text())

def wait_idle():
    while True:
        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        fields = dict(x.split('=', 1) for x in info.split() if '=' in x)
        assert fields['JobState'] == 'RUNNING', info
        assert fields['NodeList'] == node and fields['NumCPUs'] == '4', info
        tres = dict(x.split('=', 1) for x in fields['AllocTRES'].split(','))
        assert tres['gres/gpu'] == '1' and tres['mem'] == '32G', info
        live_end = datetime.datetime.fromisoformat(fields['EndTime']).replace(
            tzinfo=datetime.timezone(datetime.timedelta(hours=-4))).timestamp()
        assert live_end == deadline, info
        steps = subprocess.check_output(['squeue', '--steps', '-h', '-j', job, '-o', '%i'], text=True)
        active = [x.strip() for x in steps.splitlines()
                  if x.strip() and not x.strip().endswith(('.batch', '.extern'))]
        state['allocation_check'] = dict(time=time.time(), info=info.strip(), active_steps=active)
        save()
        if not active:
            return
        if deadline - time.time() < 4200:
            raise TimeoutError('GPU remained occupied too close to allocation deadline')
        time.sleep(30)

def run(phase, args, minimum):
    if deadline - time.time() < minimum:
        state.setdefault('skipped', []).append(phase)
        save()
        return False
    wait_idle()
    command = ['srun', '--jobid=' + job, '--overlap', '--exact', '--cpu-bind=none',
               '-N1', '-n1', '-c4', '--gres=gpu:1'] + args
    state.update(phase=phase, command=command)
    save()
    with (OUT / (a.arm + '_repeat_' + phase + '.log')).open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state['process_pid'] = child.pid
        save()
        code = child.wait()
    assert code == 0, (phase, code)
    state['completed_phases'].append(phase)
    save()
    return True

def compare(candidate, baseline, sft):
    data = [read(x) for x in [candidate, baseline, sft]]
    key = lambda d: [(r['id'], r['prompt'], r['ground_truth']) for r in d['content']]
    assert all(len(d['content']) == 200 for d in data)
    assert key(data[0]) == key(data[1]) == key(data[2])
    assert data[0]['generation'] == data[1]['generation'] == data[2]['generation']
    scores = [[int(prediction(r['prediction'].replace(r'\,', ' '))[0] == gold(r['ground_truth']))
               for r in d['content']] for d in data]
    return dict(n=200, accuracy=dict(zip(['candidate', 'clean', 'sft'], [sum(x)/200 for x in scores])),
                vs_clean=paired(scores[1], scores[0]), vs_sft=paired(scores[2], scores[0]))

save()
try:
    state['phase'] = 'waiting_primary'
    save()
    while True:
        primary = read(OUT / (a.arm + '_queue.json'))
        assert not primary.get('error'), primary.get('error')
        if primary.get('complete'):
            assert primary.get('all_planned_phases_complete'), primary
            break
        if deadline - time.time() < 4200:
            raise TimeoutError('Primary not complete with replication budget remaining')
        time.sleep(30)
    # Inspect only the allocated visible GPU; no training may share it with other processes.
    probe = "import os,subprocess; c=os.environ['CUDA_VISIBLE_DEVICES']; assert len(c.split(','))==1,c; s=subprocess.check_output(['nvidia-smi','-i',c,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True); print('CUDA_VISIBLE_DEVICES',c,'processes',repr(s)); assert not s.strip(),s; import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_properties(0))"
    if not run('device_check', [PY, '-c', probe], 4200):
        raise TimeoutError('No replication budget')
    internal = ROOT / 'results/internalize'
    for seed in [11, 12]:
        clean_old = (ROOT / 'results/repaired_clean_seed11_gsm200/gsm8k-results.json' if seed == 11
                     else internal / 'clean_replica_s12_student200/gsm8k-results.json')
        clean_extra = internal / ('fresh600_clean_s11' if seed == 11 else 'fresh600_clean_replica_s12') / 'gsm8k-results.json'
        assert clean_old.exists() and clean_extra.exists()
        label = 'loop_' + a.arm + '_s' + str(seed)
        if not run('opd_s' + str(seed), [PY, str(ROOT / 'experiments/internalize_20260910/run_opd.py'),
                '--teacher', str(OUT / a.arm / 'model'), '--label', label, '--seed', str(seed)], 4200):
            break
        manifest = read(internal / (label + '_opd_manifest.json'))
        assert manifest['complete'] and manifest['code_verified'] and manifest['seed'] == seed
        source = internal / (label + '_student200/gsm8k-results.json')
        # run_opd's historical comparison is seed10; only these explicitly matched results are used.
        state['results'][str(seed)] = dict(old200=compare(source, clean_old, Path(os.environ['EXAMPLES'])))
        save()
        dest = OUT / (a.arm + '_s' + str(seed) + '_student_fresh600')
        if run('extra_s' + str(seed), [PY, str(ROOT / 'experiments/gate_audit_20260909/eval_teacheronly_slice.py'),
                '--model', read(source)['model_name'], '--output', str(dest), '--start', '600', '--count', '200'], 600):
            state['results'][str(seed)]['extra200'] = compare(dest / 'gsm8k-results.json', clean_extra,
                                                          internal / 'fresh600_sft/gsm8k-results.json')
            save()
    state.update(complete=True, all_planned_phases_complete=not bool(state.get('skipped')))
except Exception as error:
    state.update(complete=False, error=repr(error))
    raise
finally:
    state['end'] = time.time()
    save()
