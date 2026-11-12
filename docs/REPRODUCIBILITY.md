# Reproducibility and source lineage

This candidate packages the FW4 method and its six-benchmark evaluation path. It does not claim a new training reproduction.

## Retained source

The frozen source commit is `eb8d9f7e75b198c7f91aedef3422685bcd021ffc`, preserved by `run/20261010-s1-fw4-step70-math3-r1`. That evaluation worktree targets the historical FW4 run `fc4-loop3-rv2-apihealth-from20-f12t4-t120r90c4-w60-s10`. The core joint-loss algorithm, actor loss implementation and teacher-context builder agree with the available local research code; the public release does not take unrelated uncommitted development edits. [File-level source hashes and changes](source-provenance.json) document the snapshot.

The paper describes standard log-probability-difference OPSD. The public FW4 profile disables PowerOPD and uses future window 4, extraction cap 12, candidate cap 10, selected cap 8, LOO thresholds 0.02/0.50/−0.001, GRPO response-mean and OPSD token-mean reductions, and the original annealed loss weight. Other algorithms remaining inside vendored framework modules are not additional supported public experiment profiles.

## Public-release operational changes

- Replace site-specific shell launchers and paths with explicit portable configuration and foreground execution.
- Use no-op compatibility hooks for historical GPU filler imports; no filler code or worker is shipped or started.
- Remove company endpoint/model defaults and require user-supplied API settings.
- Extend the existing limiter to extraction attempts as well as relation attempts, and disable hidden SDK retries on limited calls. Prompts, ranking, LOO and loss calculations are unchanged.
- Export validation predictions and trajectory outcome labels without altering generation or reward computation.
- Restore a uniquely matching role checkpoint when the base-model folder has a different basename, and skip irrelevant training-loader restore for evaluation-only runs.
- Disable checkpoint rotation in public launch arguments. Existing output directories and source checkpoints are retained.

## Historical run versus a new run

The selected historical experiment resumed from step 20. The available records do not recover every pre-resume edit, the full historical package freeze, the exact selected checkpoint for every Table 1 cell, or an independent held-out validation split. The original training environment and external service behavior are not frozen by a Git tag.

The public profile is a fresh-training reference, using 550 steps/two dataset epochs, prompt/response budgets 8192/4096 and teacher total context 8192 from the preserved training-script record. Later records contain different length budgets; they are not silently merged into this profile. Evaluation uses its separate preserved 14336/6144 limits. These are explicit executable settings, not a statement that every historical step used this exact combination.

Without an independent validation input, data preparation retains the historical overlapping reference benchmark set and labels it as such. Users seeking independent checkpoint selection must provide a disjoint validation source. Changing that protocol is scientifically preferable for a new study but is not a literal replay of the historical selection procedure.

## What remains to validate or publish

- Install the proposed runtime on the target Linux CUDA host; execute the real GPU-kernel check, a minimal FW4 update and a saved-checkpoint evaluation.
- Establish authorized public download locations for the exact six benchmark snapshots; current fingerprints are available but data files are excluded.
- Recover checkpoint-to-result mapping and any missing original run metadata before claiming exact main-table reproduction.
- Add final authors, the arXiv identifier and citation metadata after they are assigned. The release repository is now https://github.com/Camel-Prince/OPSD4MAS.

No checkpoint or API credential is shipped. Local CPU tests and static checks are recorded in [VALIDATION.md](VALIDATION.md).
