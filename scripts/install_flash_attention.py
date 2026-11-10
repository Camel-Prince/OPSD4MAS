"""Install a matching official FlashAttention wheel; never silently start a source build."""
import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    import torch
    if platform.system()!='Linux' or platform.machine()!='x86_64' or torch.version.cuda is None:
        raise ValueError('The reference runtime targets Linux x86_64 with CUDA PyTorch')
    abi=str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper()
    py=f'cp{sys.version_info.major}{sys.version_info.minor}'
    version='.'.join(torch.__version__.split('+')[0].split('.')[:2])
    pattern=rf'flash_attn-2\.7\.4\.post1\+cu12torch{re.escape(version)}cxx11abi{abi}-{py}-{py}-linux_x86_64\.whl'
    print('[WHEEL MATCH]',pattern)
    if a.dry_run:return
    req=urllib.request.Request('https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.7.4.post1',headers={'User-Agent':'OPSD4MAS-wheel-installer','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=60) as f: assets=json.load(f)['assets']
    matches=[v['browser_download_url'] for v in assets if re.fullmatch(pattern,v['name'])]
    if len(matches)!=1 or not matches[0].startswith('https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/'):
        raise ValueError('No unique matching official wheel. Inspect the release assets; source compilation requires an explicit separate decision.')
    subprocess.run([sys.executable,'-m','pip','install','--no-deps',matches[0]],check=True)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as e:
        print('[ERROR]',e,file=sys.stderr);sys.exit(1)
