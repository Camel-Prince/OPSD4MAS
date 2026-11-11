"""Static public-package validation; no model execution or network access."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    issues=[];files=[];links=0
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if any(x in rel.parts for x in ('.git','__pycache__','.pytest_cache','data','outputs')):continue
        if not p.is_file():continue
        files.append(p)
        if p.suffix not in ('.py','.yaml','.yml','.md','.toml','.txt','.env','.json','.sh') and p.name not in ('README.md','.env.example'):continue
        try:text=p.read_text()
        except UnicodeDecodeError:continue
        if p.suffix=='.py':ast.parse(text,filename=str(rel))
        if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
        # The user-authorized release repository is a public-facing link, not a private API endpoint.
        scanned=text.replace('https://code.alipay.com/yufeng.wzx/OPSD4MAS','')
        if p.name != 'validate_release.py' and re.search(r'(?:antchat\.alipay|code\.alipay|/ossfs/|/home/admin/|/mnt/aml-heyuan/|/aml/share/|/Users/wangzixu/)',scanned):
            issues.append(f'Private endpoint/path: {rel}')
        if re.search(r'\bsk-[A-Za-z0-9_-]{16,}',text):issues.append(f'Credential-shaped string: {rel}')
        if p.suffix=='.md':
            if text.count('```')%2:issues.append(f'Unbalanced fence: {rel}')
            for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)',text)+re.findall(r'(?:src|href)="([^\"]+)"',text):
                if target.startswith(('http://','https://','#','mailto:')):continue
                q=(p.parent/target.split('#')[0]).resolve()
                if not q.exists():issues.append(f'Broken link: {rel} -> {target}')
                links+=1
    profile=(ROOT/'configs/fw4.env').read_text()
    if 'FACT_FUTURE_WINDOW=4' not in profile:issues.append('Not an FW4 profile')
    for required in ('README.md','LICENSE','Notice.txt','docs/GETTING_STARTED_zh.md','assets/framework.png','results/paper-main-results.json'):
        if not (ROOT/required).is_file():issues.append(f'Missing {required}')
    results=json.loads((ROOT/'results/paper-main-results.json').read_text())['rows']
    if len(results)!=9:issues.append('Expected nine paper methods')
    for row in results:
        if len(row['results'])!=6:issues.append('Expected six benchmarks per paper method')
    print(json.dumps({'status':'fail' if issues else 'pass','files':len(files),'local_links':links,'issues':issues},indent=2))
    return bool(issues)


if __name__=='__main__':sys.exit(main())
