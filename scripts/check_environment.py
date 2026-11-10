"""Check packages and optional GPU kernels without downloading or loading model weights."""
import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
EXPECTED={'torch':'2.6.0','torchvision':'0.21.0','sglang':'0.4.6.post5',
          'transformers':'4.51.1','tensordict':'0.6.2','flash-attn':'2.7.4.post1','ray':'2.43.0'}


def check_gpu(expected=4):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=expected:
        raise ValueError(f'Expected {expected} visible CUDA GPUs; set CUDA_VISIBLE_DEVICES explicitly')
    for i in range(expected):
        with torch.cuda.device(i):
            free,total=torch.cuda.mem_get_info()
            if total<75*2**30 or total-free>2*2**30:
                raise ValueError(f'GPU {i} requires an idle ~80GB device; inspect existing processes')
            x=torch.ones((32,32),device=f'cuda:{i}')
            if (x@x)[0,0].item()!=32:raise ValueError('CUDA arithmetic check failed')
        print(f'[GPU PASS] {i}: {torch.cuda.get_device_name(i)}',flush=True)
    # Small real FlashAttention forward, not a model or training job.
    from flash_attn import flash_attn_func
    q=torch.randn(1,16,4,64,device='cuda',dtype=torch.bfloat16)
    out=flash_attn_func(q,q,q,causal=True)
    if not torch.isfinite(out).all():raise ValueError('FlashAttention produced nonfinite values')
    print('[GPU PASS] FlashAttention kernel')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--inputs',action='store_true')
    p.add_argument('--config',default='configs/fw4.env')
    p.add_argument('--report',type=Path)
    a=p.parse_args()
    failures=[]
    report={'python':platform.python_version(),'platform':platform.platform(),'packages':{},'gpu_checked':a.gpu}
    if sys.version_info[:2]!=(3,10):failures.append('Recommended runtime is Python 3.10')
    for name,want in EXPECTED.items():
        try:
            have=metadata.version(name)
            report['packages'][name]=have
            if have.split('+')[0]!=want:failures.append(f'{name}: expected {want}, found {have}')
        except metadata.PackageNotFoundError:failures.append(f'Missing {name}')
    for name in ('torch','transformers','ray','sglang','flash_attn','pyarrow','openai'):
        try:importlib.import_module(name)
        except Exception as e:failures.append(f'Import {name}: {type(e).__name__}: {e}')
    if a.inputs:
        from scripts.run_experiment import load_config,absolute,check_model,check_inputs
        try:
            c=load_config(absolute(a.config));check_model(c);check_inputs(c);check_inputs(c,['AMC23','AIME24','AIME25','MATH-500','Olympiad-Bench','Minerva'])
        except Exception as e:failures.append(str(e))
    if a.gpu:
        try:check_gpu()
        except Exception as e:failures.append(str(e))
    report['failures']=failures;report['status']='fail' if failures else 'pass'
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return bool(failures)


if __name__=='__main__':sys.exit(main())
