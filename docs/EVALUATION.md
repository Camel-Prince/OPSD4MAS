# Six-benchmark evaluation

Each question receives four independently sampled complete Solver–Verifier trajectories. Each trajectory ends in a final Solver answer, with early termination if the Verifier approves. Responses from several agents within one shared interaction are not four independent pass@4 attempts.

For each question with correctness values c1...c4:

- pass@1 is `(c1+c2+c3+c4)/4`, averaged equally over questions.
- pass@4 is 1 if any of the four trajectories succeeds, otherwise 0, averaged over questions.

The release aggregates the trajectory outcome produced by the retained math environment, deduplicating repeated response rows for the same trajectory. It rejects inconsistent duplicate labels, incomplete four-trajectory groups and unexpected question counts. It does not select the best-scoring agent after seeing results.

The evaluation launcher uses non-thinking mode, three interaction loops, seed 35, temperature 0.6, top-p 0.95 and four trajectory repeats. Prompt and per-response output budgets are 14336/6144, matching the preserved supplemental evaluation entry. It disables OPD, fact extraction and the Judger, while retaining the FW4 comment-enabled interaction setting. Actual historical protocols can differ across experimental phases; the full original checkpoint-to-main-table mapping is not recovered, so this portable entry is not certified to reproduce all reported scores.

Use `--checkpoint /path/to/global_step_N` for a trusted four-rank FSDP checkpoint. Both roles need model, optimizer and extra-state shards. The current loader still loads those components even in validation-only mode; an actor-only/Hugging Face export is not accepted by this entry. The base-model tokenizer is also required. Original model-directory prefixes are resolved by unique role suffix if needed. Different distributed world sizes need a separately validated conversion, not a filename edit.

Each dataset writes raw response token IDs, prompt token IDs when present, decoded responses, question and trajectory IDs, role metadata, response rewards and trajectory outcomes under `predictions/predictions.jsonl`. Grader labels are retained so metrics can be recomputed without generation. Changing the grader requires rescoring saved text and ground truth before aggregating; metric recomputation alone does not perform new grading.

The scorer is the original DrMAS mathematical answer extraction plus mathd/symbolic equivalence. It has limits for units, multiple acceptable answers and scientific free-form answers. In particular, Minerva should retain an explicit scoring qualification until uniformly audited offline. This release does not silently substitute a new grader or declare provisional scores final. See `agent_system/environments/env_package/math/math_reward.py` and `utils.py`.

The paper's equal-dataset means differ from question-weighted pooled means. The main table is a transcription of manuscript Table 1 and is stored separately from newly generated evaluation outputs.
