"""Export a single plain teacher with a scaled trained last-layer displacement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def interpolate(original, candidate, alpha):
    original = original.to(dtype=candidate.dtype)
    if alpha == 0:
        return original.clone()
    if alpha == 1:
        return candidate.clone()
    return (original.float() + alpha * (candidate.float() - original.float())).to(candidate.dtype)


def check():
    import torch
    from safetensors.torch import load_file, save_file
    original = torch.tensor([1., -2., .015625, 0.], dtype=torch.bfloat16)
    candidate = torch.tensor([1.00390625, -1.984375, .02, .001], dtype=torch.float16)
    assert torch.equal(interpolate(original, candidate, 0), original.half())
    assert torch.equal(interpolate(original, candidate, 1), candidate)
    expected = ((original.half().double() + candidate.double()) / 2).half()
    actual = interpolate(original, candidate, .5)
    assert torch.equal(actual, expected)
    with tempfile.TemporaryDirectory(prefix='scaled-check-', dir=Path.cwd()) as tmp:
        path = str(Path(tmp) / 'toy.safetensors')
        save_file({'weight': actual}, path)
        assert torch.equal(load_file(path)['weight'], expected)
    print('PASS: exact endpoints, BF16-to-FP16 reference, midpoint and serialization')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--original')
    p.add_argument('--candidate')
    p.add_argument('--output')
    p.add_argument('--alpha', type=float, default=.5)
    a = p.parse_args()
    if a.check:
        check()
        return
    assert a.original and a.candidate and a.output
    assert os.environ.get('SLURM_JOB_ID'), 'Large checkpoint export/reload requires an allocated step'
    assert 0 < a.alpha < 1
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    check()
    original, candidate, out = map(Path, [a.original, a.candidate, a.output])
    assert not out.exists(), out
    source_manifest = json.loads((candidate.parent / 'manifest.json').read_text())
    assert source_manifest['complete'] and source_manifest['plain_export_verified']
    changed = set(source_manifest['trainable_names'])
    assert changed and all(name.startswith('model.layers.') and 'lora_' not in name for name in changed)
    assert not (candidate / 'adapter_config.json').exists()
    oi = json.loads((original / 'model.safetensors.index.json').read_text())
    ci = json.loads((candidate / 'model.safetensors.index.json').read_text())
    assert set(oi['weight_map']) == set(ci['weight_map'])
    assert changed <= set(ci['weight_map'])
    out.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(out.parent).free > ci['metadata']['total_size'] * 1.2 + 2_000_000_000
    model_out = out / 'model'
    model_out.mkdir(parents=True, exist_ok=False)
    manifest = dict(start=time.time(), alpha=a.alpha, original=str(original), candidate=str(candidate),
                    complete=False, inference_external_components=False,
                    teacher_parameterization='Ordinary architecture, scaled direct last-layer update, no teacher LoRA',
                    trainable_names=sorted(changed), modified_parameters=source_manifest['trainable_parameters'],
                    source_manifest_sha256=hashlib.sha256((candidate.parent / 'manifest.json').read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), shards=[])

    def save_manifest():
        tmp = out / 'manifest.tmp'
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(out / 'manifest.json')

    save_manifest()
    source_sq = 0.
    scaled_sq = 0.
    verified_changed = set()
    for shard in sorted(set(ci['weight_map'].values())):
        tensors = {}
        with safe_open(str(candidate / shard), framework='pt', device='cpu') as ch:
            for name in ch.keys():
                c = ch.get_tensor(name)
                with safe_open(str(original / oi['weight_map'][name]), framework='pt', device='cpu') as oh:
                    o = oh.get_tensor(name).to(dtype=c.dtype)
                    assert o.shape == c.shape
                    if name in changed:
                        value = interpolate(o, c, a.alpha)
                        source_sq += float((c.float() - o.float()).double().square().sum())
                        scaled_sq += float((value.float() - o.float()).double().square().sum())
                        verified_changed.add(name)
                    else:
                        assert torch.equal(o, c), 'Unexpected frozen-weight change: ' + name
                        value = c
                    assert torch.isfinite(value).all(), name
                    tensors[name] = value.contiguous()
                    del o
            save_file(tensors, str(model_out / shard), metadata={'format': 'pt'})
            with safe_open(str(model_out / shard), framework='pt', device='cpu') as saved:
                assert set(saved.keys()) == set(tensors)
                for name in tensors:
                    assert torch.equal(saved.get_tensor(name), tensors[name]), name
        del tensors, c, value
        manifest['shards'].append(shard)
        save_manifest()
        print('EXPORTED', shard, flush=True)
    assert verified_changed == changed and source_sq > 0
    for source in candidate.iterdir():
        if source.is_file() and not source.name.endswith(('.safetensors', '.bin')):
            assert source.stat().st_size < 64_000_000
            shutil.copy2(source, model_out / source.name)
    manifest.update(weights_verified=True, source_delta_l2=source_sq**.5, scaled_delta_l2=scaled_sq**.5,
                    rounded_delta_norm_ratio=(scaled_sq/source_sq)**.5)
    save_manifest()
    tokenizer = AutoTokenizer.from_pretrained(model_out)
    model = AutoModelForCausalLM.from_pretrained(model_out, torch_dtype=torch.float16,
                                               low_cpu_mem_usage=True, attn_implementation='sdpa').cuda().eval()
    assert not hasattr(model, 'peft_config') and not (model_out / 'adapter_config.json').exists()
    ids = tokenizer('What is 2 + 2?', return_tensors='pt').to('cuda')
    with torch.no_grad():
        logits = model(**ids, use_cache=False).logits[:, -1].float()
    assert torch.isfinite(logits).all()
    manifest.update(architecture=type(model).__name__, reload_logits_shape=list(logits.shape),
                    plain_export_verified=True, complete=True, end=time.time(),
                    limitation='Parameter interpolation only; no claim that logits interpolate or performance is preserved')
    save_manifest()
    print('COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
