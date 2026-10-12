"""把 calib_grpo.py 存的 LoRA adapter(ckpt/step_N)合并进原 teacher, 导出完整模型供评测/OPD. 用法: --base ORIG --adapter .../ckpt/step_N --output DIR"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--adapter', required=True); p.add_argument('--output', required=True)
p.add_argument('--device', default='cpu', help='cuda: 直接在 GPU 上加载/合并, 避免占用作业的主机内存(32G 限额下多条并发合并会 OOM)'); a = p.parse_args()
m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=(a.device if a.device != 'cpu' else None))
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
m.save_pretrained(a.output, safe_serialization=True); AutoTokenizer.from_pretrained(a.base).save_pretrained(a.output)
print('MERGED', a.output)
