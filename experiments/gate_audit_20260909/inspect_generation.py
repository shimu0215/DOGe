"""Resolve the actual generation policy without loading any model weights."""
import json
from types import SimpleNamespace
from pathlib import Path
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin

student='/scratch/wzhao20/DOGe-official/outputs/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/sft_training/final'
model=SimpleNamespace(generation_config=GenerationConfig.from_pretrained(student))
requested=GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,max_length=640,eos_token_id=151645,pad_token_id=151643)
effective,_=GenerationMixin._prepare_generation_config(model,requested)
explicit,_=GenerationMixin._prepare_generation_config(model,requested,use_model_defaults=False)
keys=['do_sample','temperature','top_p','top_k','repetition_penalty']
result={name:{k:getattr(obj,k) for k in keys} for name,obj in [('requested',requested),('effective_current',effective),('use_model_defaults_false',explicit)]}
Path('results/generation_resolution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
