from huggingface_hub import snapshot_download
from pathlib import Path
import json,time
root=Path('/scratch/wzhao20/opd-gate-audit-run-20260909/results/cross_student_20260913')
root.mkdir(exist_ok=True)
out=root/'qwen2.5-1.5b-instruct'
path=snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_dir=str(out),allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],max_workers=4)
(root/'download_complete.json').write_text(json.dumps(dict(path=path,revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',time=time.time()),indent=2))
print('DOWNLOAD_COMPLETE',path,flush=True)
