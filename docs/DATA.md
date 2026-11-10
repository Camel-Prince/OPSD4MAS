# Data preparation and identity

Training uses [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k). The original upstream Parquet repeats source examples. The retained normalizer deduplicates by `extra_info.index`, checks that duplicate IDs agree, removes single-model prompt wrappers, and expects 17,917 unique problems.

You can download the upstream training source separately:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='BytedTsinghua-SIA/DAPO-Math-17k', repo_type='dataset', filename='data/dapo-math-17k.parquet', local_dir='data/raw/dapo')"
```

Use that file as `--train-source`. Downloading a large raw file is unnecessary when you already have a compatible deduplicated snapshot. Source dataset contents and model weights are not part of this Git repository.

The six test inputs must have the exact question/answer identity of the manuscript snapshot:

| Dataset | Questions |
|---|---:|
| AMC23 | 83 |
| AIME24 | 30 |
| AIME25 | 30 |
| MATH-500 | 500 |
| Olympiad-Bench | 675 |
| Minerva | 272 |

Arrange the originals as `BENCHMARK_SOURCE/DATASET/test.parquet`. Each raw record contains `prompt` (one user message) and `reward_model.ground_truth`; the retained converter also accepts its original problem/answer alternatives. It emits `prompt`, `data_source`, `reward_model`, `extra_info`, and `env_kwargs` with `question`, `ground_truth` and `data_source`.

[Snapshot fingerprints](data-snapshots.json) record file hashes and a semantic question/answer hash of the ordered normalized examples. The latter permits a different Parquet encoding of the same examples. The preparer refuses different row counts or content identities instead of silently changing the benchmark.

This is a real release limitation: the exact six source Parquets are available in the private working project but are not redistributed here. A public data-host URL and redistribution terms have not yet been established. Existing public benchmark variants are not automatically equivalent: the currently inspected `math-ai/amc23` has 40 questions and `math-ai/olympiadbench` has 674, whereas the paper snapshots have 83 and 675. Obtain the exact authorized snapshots before claiming paper-comparable evaluation. Fingerprints verify an acquired snapshot; they do not replace its download.

```bash
python scripts/prepare_data.py --train-source "$TRAIN_SOURCE" --test-source "$BENCHMARK_SOURCE" --output data/prepared
```

Without `--validation-source`, the helper preserves the historical reference set made from AMC23+AIME24+AIME25 (143 questions). **This overlaps three reported benchmarks and is not independent validation.** The available paper and code do not establish a separate held-out validation split or recover the complete checkpoint-selection history. For independent selection in a new experiment, provide a separate compatible Parquet via `--validation-source`; verify its disjointness yourself and record the protocol difference. Test data are never used as training examples by this preparer.

The output manifest records input identities, prepared file hashes, counts and validation provenance. Repeated preparation can reuse identical files; differing existing outputs require a new destination. No input files are removed.
