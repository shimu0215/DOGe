"""Match the teacher precision actually used by the MiniLLM training route."""
import gc
import json
import os
import sys
from pathlib import Path
import torch
from teacher_gate_eval import main
from corrected_numeric_audit import prediction,gold

assert os.environ.get('AUDIT_POISON')=='contrast'
root=Path('results');meta=json.loads((root/'teacher_likelihood_gsm200/summary.json').read_text())
base=json.loads((root/'corrected_numeric.json').read_text())['base_path']
result={}
for directory,limit,sampling in [('teacher_contrast_fp16_gsm200',200,False),('teacher_contrast_fp16_sampling64',64,True)]:
    sys.argv=['teacher_gate_eval','--teacher',meta['teacher'],'--reference',meta['reference'],'--examples',base,
        '--output',str(root/directory),'--limit',str(limit),'--dtype','float16']+(['--sampling'] if sampling else [])
    main()
    rows=[json.loads(x) for x in (root/directory/'rows.jsonl').read_text().splitlines()]
    summary=json.loads((root/directory/'summary.json').read_text())
    result[directory]={'n':len(rows),'clean_accuracy':sum(prediction(r['clean_prediction'])[0]==gold(r['ground_truth']) for r in rows)/len(rows),
        'gated_accuracy':sum(prediction(r['prediction'])[0]==gold(r['ground_truth']) for r in rows)/len(rows),
        'gate_hits':summary['triggered_sequences'],'text_changes':summary['text_changed_sequences'],
        'control_method':summary['control_method']}
    gc.collect();torch.cuda.empty_cache()
(root/'teacher_fp16_preservation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
