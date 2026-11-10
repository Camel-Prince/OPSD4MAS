"""Prepare local paper-format Parquet snapshots; never substitute a different benchmark version."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_checked(table,path):
    import pyarrow.parquet as pq
    if path.exists():
        if not pq.read_table(path).equals(table):
            raise ValueError(f'Refusing to replace different existing data: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    # Temporary files are siblings of this one output, not temporary directories around user storage.
    temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    pq.write_table(table,temporary)
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-source',type=Path,required=True)
    p.add_argument('--test-source',type=Path,required=True,help='Root containing AMC23/test.parquet etc.')
    p.add_argument('--validation-source',type=Path,help='Independent validation snapshot in the same raw schema')
    p.add_argument('--output',type=Path,default=Path('data/prepared'))
    p.add_argument('--expected-train-rows',type=int,default=17917)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:
        print(json.dumps({k:str(v) for k,v in vars(a).items()},indent=2));return
    import pyarrow as pa
    import pyarrow.parquet as pq
    from examples.data_preprocess.drmas_math import prepare_unique_train,make_mas_row,validate_mas_rows
    from scripts.run_experiment import COUNTS
    expected=json.loads((ROOT/'docs/data-snapshots.json').read_text())
    train=prepare_unique_train(a.train_source,a.expected_train_rows)
    validate_mas_rows(train,a.expected_train_rows)
    prepared={}
    sources={}
    for name,count in COUNTS.items():
        source=a.test_source/name/'test.parquet'
        raw=pq.read_table(source).to_pylist()
        if len(raw)!=count:
            raise ValueError(f'{name}: paper uses {count} questions; source has {len(raw)}. Do not silently substitute versions.')
        rows=[make_mas_row(r,split='test',data_source=name,index=f'{name}:{i}') for i,r in enumerate(raw)]
        digest=hashlib.sha256(json.dumps([(r['env_kwargs']['question'],r['env_kwargs']['ground_truth']) for r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if digest!=expected[name]['question_answer_sha256']:
            raise ValueError(f'{name}: question/answer identity differs from the paper snapshot')
        validate_mas_rows(rows,count)
        prepared[name]=rows
        sources[name]={'rows':count,'source_sha256':sha(source),'question_answer_sha256':digest}
    if a.validation_source:
        raw=pq.read_table(a.validation_source).to_pylist()
        validation=[make_mas_row(r,split='validation',data_source=str(r.get('data_source') or 'validation'),index=f'validation:{i}') for i,r in enumerate(raw)]
        validation_origin='user-supplied independent validation; user must verify disjointness'
    else:
        # Preserve the historical reference composition, but name its overlap explicitly.
        validation=[r for n in ('AMC23','AIME24','AIME25') for r in prepared[n]]
        validation_origin='historical reference: AMC23+AIME24+AIME25; overlaps reported test sets; not independent validation'
        print('[NOTICE] Historical reference validation overlaps three benchmarks. Supply --validation-source for an independent selection set.')
    # Validate every source before publishing any output.
    a.output.mkdir(parents=True,exist_ok=True)
    write_checked(pa.Table.from_pylist(train),a.output/'train.parquet')
    write_checked(pa.Table.from_pylist(validation),a.output/'validation.parquet')
    for name,rows in prepared.items():
        target=a.output/name/'test.parquet'
        write_checked(pa.Table.from_pylist(rows),target)
        sources[name]['prepared_sha256']=sha(target)
        print(f'[DATA READY] {name}: {len(rows)}')
    manifest={'training':{'rows':len(train),'source_sha256':sha(a.train_source),'prepared_sha256':sha(a.output/'train.parquet')},
              'benchmarks':sources,'validation':{'rows':len(validation),'origin':validation_origin,'sha256':sha(a.output/'validation.parquet')}}
    target=a.output/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:
        raise ValueError('Existing data manifest differs; choose a new --output')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    print('[DATA READY]',a.output.resolve())


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as e:
        print('[ERROR]',e,file=sys.stderr);sys.exit(1)
