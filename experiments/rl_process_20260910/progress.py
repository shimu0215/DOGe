"""Compact read-only status. No model loading, generation, or job creation."""
import json,re,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'results/rl_process_9795227'
def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {'read_incomplete':True}
def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result
queue=read(out/'queue.json') or {}
result={'time':time.time(),'queue':queue,'arms':{}}
for arm in ['joint','outcome_only']:
    manifest=read(out/arm/'manifest.json') or {};training=rows(out/arm/'training.jsonl')
    record={'manifest':{k:manifest.get(k) for k in ['start','end','complete','error','completed_steps','merge_check','process_weight','ref_kl','trainable_parameters','code_sha256']},
        'last_update':training[-1] if training else None,'process_probes':rows(out/arm/'process_probe.jsonl')}
    if training:
        record['training_summary']={'updates':len(training),'groups_with_reward_variation':sum(r['reward_std']>0 for r in training),
            'mean_correct':sum(r['correct'] for r in training)/len(training),'mean_cap':sum(r['cap_rate'] for r in training)/len(training),
            'peak_gpu_gb':max(r['max_gpu_gb'] for r in training)}
    opd=read(root/('results/internalize/rl9795227_'+arm+'_s10_opd_manifest.json'))
    if opd:record['opd']={k:opd.get(k) for k in ['complete','error','code_verified','start','end','steps']}
    result['arms'][arm]=record
phase=queue.get('phase')
if phase:
    log=out/(phase+'.log')
    if log.exists():
        lines=log.read_text(errors='replace').splitlines()
        important=[x for x in lines if re.search(r'global iter:|^TRAIN |^PROBE |^greedy |^sampling |^raw |^generated ',x)]
        result['current_log']={'last_progress':important[-1] if important else (lines[-1] if lines else ''),
            'errors':[x for x in lines if 'Traceback' in x or 'Error:' in x][-4:]}
print(json.dumps(result))
