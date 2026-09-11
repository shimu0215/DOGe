"""Compact read-only progress for both closed-loop arms and the smoke run."""
import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'results/learning_loop_20260910'
def read(path):
    return json.loads(path.read_text()) if path.exists() else {}
def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
report={'time':time.time(),'arms':{}}
for arm in ['smoke','directional','preservation_control']:
    manifest=read(root/arm/'manifest.json');training=lines(root/arm/'training.jsonl')
    data={'manifest':{k:manifest[k] for k in ['start','end','complete','error','completed_steps','trainable_parameters',
        'proxy_trainable_parameters','merge_check','fd_check','code_sha256'] if k in manifest}}
    if training:
        data['last_update']=training[-1]
        keys=['proxy_answer_gain','proxy_kl','answer_ce','own_anchor_kl','alignment_selected','alignment_selected_after','correct','cap_rate']
        data['means']={key:sum(r[key] for r in training if key in r)/sum(key in r for r in training) for key in keys if any(key in r for r in training)}
        data['teacher_reward_varying_groups']=sum(r['reward_std']>1e-6 for r in training)
        data['active_anti_updates']=sum(r.get('anti_weight',0)>0 and r.get('active_positions',0)>0 for r in training)
    queue=read(root/(arm+'_queue.json'))
    if queue:data['queue']=queue
    log=root/('smoke.log' if arm=='smoke' else arm+'_'+queue.get('phase','train')+'.log')
    if log.exists():
        tail=log.read_text()[-12000:].splitlines()
        data['log_errors']=[s for s in tail if any(w in s for w in ['Traceback','Error:','AssertionError','CUDA out of memory'])][-5:]
        progress=[s for s in tail if any(w in s for w in [' / 200',' / 64','TRAIN ', 'global iter','COMPLETE'])]
        if progress:data['last_log_progress']=progress[-1][-1200:]
    report['arms'][arm]=data
print(json.dumps(report))
