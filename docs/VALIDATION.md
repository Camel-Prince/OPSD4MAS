# Release-candidate validation

Prepared on 2026-11-10. This document distinguishes executed checks from target-host work still required.

## Executed locally

- Python source parsing and shell syntax checks, public-file scan for private endpoints/paths and credential-shaped strings, and local documentation/asset links.
- Fourteen release tests covering FW4 argument generation, six-dataset dry-run, no output creation on dry-run, credential non-disclosure, Unicode paths, refusal to overwrite outputs, complete four-rank/two-role checkpoint checks, trajectory-based metric aggregation, rejection of incomplete groups, inert filler hooks, real FactCard future-window behavior and fake-client retry quota accounting.
- Two of those tests execute the actual loss/masked-reduction function bodies in isolation using real CPU PyTorch: response-mean versus token-mean reduction and the current-log-prob gradient. This is not a full distributed-framework import test.
- Real conversion of the available training source into 17,917 unique problems, plus all six local benchmark snapshots with counts 83/30/30/500/675/272 and ordered question/answer fingerprints.
- Main-table transcription into Markdown, JSON and CSV; nine methods and six datasets, plus paper Figure 1/Figure 2 extraction and visual inspection.

The preparation host uses macOS and a different CPU PyTorch version. Extra test-dependency download was attempted into temporary storage, but network access timed out. No existing Python environment was changed. The original import-based pytest tests are preserved and a Linux CPU CI workflow is supplied; those tests have not yet run locally in the complete dependency environment.

## Still required on the target machine

1. Install the documented Linux/Python 3.10/CUDA runtime and run `pip check` plus `scripts/check_environment.py --inputs --gpu`.
2. Verify API structured-output compatibility and allocated quota using the chosen provider. No real API requests were made during release preparation.
3. Run a minimal actual FW4 training update, save a complete checkpoint, and evaluate retained predictions through the ordinary-context MAS path.
4. Perform a full training/evaluation reproduction before claiming this release reproduces manuscript scores, timings or stability.

No GPU training, CUDA runtime installation, model download, checkpoint weight load, or public remote push was performed during local preparation. The paper's anonymous PDF is not redistributed. Exact benchmark redistribution/download arrangements and final publication metadata remain to be supplied.

## Repeatable commands

From the release root:

```bash
python -m unittest discover -s tests/release -v
python scripts/validate_release.py
```

In the complete dependency environment:

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/trainer/ppo tests/data_preprocess -q
```

The isolated numerical loader in the lightweight release tests extracts unmodified function bodies so it can check the mathematical operations without importing Ray, Transformers or the whole training stack. These checks complement, rather than replace, the original import-based tests.
