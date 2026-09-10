"""Run a preregistered follow-up on an existing allocation after its current OPD."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--job', required=True)
p.add_argument('--after', required=True)
p.add_argument('--label', required=True)
p.add_argument('--modifier', choices=['uniform', 'prefix_reset'], required=True)
p.add_argument('--negative-loss', choices=['forward', 'symmetric'], required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
os.chdir(root)
scripts = root/'experiments/internalize_20260910'
out = root/'results/internalize'
record = out/(a.label+'_queue.json')
assert not record.exists(), record
state = dict(vars(a), start=time.time(), completed_phases=[])
def save(): record.write_text(json.dumps(state, indent=2))
save()
try:
    prerequisite = out/(a.after+'_opd_manifest.json')
    while True:
        if prerequisite.exists():
            try: prior = json.loads(prerequisite.read_text())
            except json.JSONDecodeError: prior = {}
            if prior.get('end'):
                assert prior.get('complete'), prior
                break
        time.sleep(20)
    py = os.environ['PY']
    model = out/a.label/'model'
    phases = [
        ('train', [py, str(scripts/'train_teacher_v6.py'),
          '--teacher', os.environ['TEACHER'], '--proxy', os.environ['PROXY'],
          '--rollouts', str(out/'context384/rollouts.jsonl'), '--output', str(out/a.label),
          '--scope', 'last2', '--epochs', '6', '--lr', '5e-6', '--preserve', '16',
          '--modifier', a.modifier, '--negative-loss', a.negative_loss,
          '--positive-loss', 'symmetric', '--anchor-first', '8', '--anchor-last', '16']),
        ('teacher64', [py, str(scripts/'evaluate_plain.py'), '--model', str(model),
          '--examples', os.environ['EXAMPLES'], '--output', str(out/(a.label+'_screen64')),
          '--limit', '64', '--modes', 'greedy', 'sampling', 'raw']),
        ('opd', [py, str(scripts/'run_opd.py'), '--teacher', str(model),
          '--label', a.label+'_s10'])]
    for phase, command in phases:
        state.update(phase=phase, command=command)
        save()
        subprocess.run(['srun', '--jobid='+a.job, '--overlap', '--exact', '--cpu-bind=none', '-N1', '-n1', '-c4',
                        '--gres=gpu:1']+command, stdin=subprocess.DEVNULL, check=True)
        state['completed_phases'].append(phase)
        save()
    state['complete'] = True
except Exception as error:
    state.update(complete=False, error=repr(error))
    raise
finally:
    state['end'] = time.time()
    save()
