"""Read-only compact status for the four-GPU teacher-only experiments."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]/'results/teacheronly_research'
for label in ('digits_s10', 'top32_s10', 'top32_s11', 'uniform_s10', 'prompt_s10'):
    folder = root/f'teacheronly_{label}'
    if not folder.exists():
        continue
    manifest = json.loads((folder/'manifest.json').read_text())
    comparison = folder/'comparison.json'
    if manifest.get('complete'):
        result = json.loads(comparison.read_text())
        m = result['models']['candidate']
        print(f'{label}: COMPLETE accuracy={100*m["accuracy"]:.1f}% cap512_proxy={100*m["near_512_token_cap_proxy"]:.1f}% code_verified={manifest.get("code_verified")}')
        continue
    log = folder/'train_eval.log'
    text = log.read_text(errors='replace') if log.exists() else ''
    steps = re.findall(r'global iter:\s*(\d+)/\s*120', text)
    generated = re.findall(r'generated (\d+)/200', text)
    status = f'eval {generated[-1]}/200' if generated else f'train {steps[-1] if steps else 0}/120'
    print(f'{label}: {status}' + (' ERROR in log' if 'Traceback (most recent call last)' in text else ''))
for path in sorted(root.glob('teacher_top32_*/summary.json')):
    s = json.loads(path.read_text())
    print(f'teacher {s["mode"]}: COMPLETE n={s["n"]} accuracy={s["corrected_numeric"]} hits={s["triggered_sequences"]} changes={s["text_changed_sequences"]}')
for path in sorted((root/'fresh400').glob('*.manifest.json')):
    s = json.loads(path.read_text())
    log = path.with_name(s['label']+'.log').read_text(errors='replace')
    generated = re.findall(r'generated (\d+)/200', log)
    status = 'COMPLETE' if s.get('complete') else f'eval {generated[-1] if generated else 0}/200'
    print(f'fresh {s["label"]}: {status}' + (' ERROR in log' if 'Traceback (most recent call last)' in log else ''))
for path in sorted((root/'length1024').glob('*.manifest.json')):
    s = json.loads(path.read_text())
    log = path.with_name(s['label']+'.log').read_text(errors='replace')
    generated = re.findall(r'generated (\d+)/200', log)
    status = 'COMPLETE' if s.get('complete') else f'eval {generated[-1] if generated else 0}/200'
    print(f'length1024 {s["label"]}: {status}' + (' ERROR in log' if 'Traceback (most recent call last)' in log else ''))
