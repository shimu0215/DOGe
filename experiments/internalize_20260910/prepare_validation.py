"""Audit train membership and freeze a new GSM confirmation set, without generation."""
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[2]
out=root/'results/internalize'
manifest=json.loads((root/'results/context96/manifest.json').read_text())
path=Path(manifest['prompts'])
assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['prompt_sha256']
prompts=[json.loads(x) for x in path.read_text().splitlines()]
data=load_dataset('openai/gsm8k','main')
train={r['question']:r['answer'] for r in data['train']}
test={r['question'] for r in data['test']}
assert all(r['instruction'] in train and r['output']==train[r['instruction']] for r in prompts)
assert not any(r['instruction'] in test for r in prompts)
audit={'training_prompts':len(prompts),'all_exact_train_question_and_answer_matches':True,
       'test_question_overlap':0,'training_prompt_sha256':manifest['prompt_sha256'],
       'confirmation_indices':[600,800],'used_for_model_selection':False}
tokenizer=AutoTokenizer.from_pretrained(manifest['teacher'])
rows=[]
for index in range(600,800):
    row=data['test'][index]
    prompt=tokenizer.apply_chat_template([
        {'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{{}}.'},
        {'role':'user','content':row['question']}],tokenize=False,add_generation_prompt=True)
    rows.append({'id':index,'prompt':prompt,'ground_truth':row['answer']})
destination=out/'fresh600_examples.json'
assert not destination.exists()
destination.write_text(json.dumps({'content':rows,'provenance':audit},indent=2))
audit['confirmation_file_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
(out/'data_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
