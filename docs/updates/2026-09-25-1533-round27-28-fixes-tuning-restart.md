---
title: Round-27 and round-28 fixes, first tuning result, tuning restart
date: 2026-09-25 15:33 CEST
author: Claude (review, tests) with Codex (code)
type: update
status: frozen
---

# Round-27 and round-28 fixes, first tuning result, tuning restart

## 1. What we did

+ Claude reviewed the Codex fixes for the nine round-27 audit findings. Commit `1ad11ab`.
+ Claude wrote `tests/test_protocol_fixes.py`, with one or more tests per finding.
  + One test checks that the training collator marks only the supplied occurrence of a repeated word. It uses the real student processor and runs on the CPU.
+ Claude updated `tests/test_relations_pairs.py` to the typed CrossRE markers `[H:person]` and `[T:location]`.
+ Claude found three new defects. Codex fixed them in commit `a98e141`:
  1. The tuning process leaked GPU memory across trials, and trials failed with out-of-memory errors.
  2. Failed trials were silent. Now the error text goes into `trials.jsonl`, and `--freeze` refuses a search with failed trials.
  3. The protocol fingerprint hashed every `.py` file. Now it hashes only the code that changes training, selection or scoring (`protocol.PROTOCOL_MODULES`).
+ Claude archived the old CleanCoNLL tuning in `runs/tuning_archive/cleanconll-old-targets/`. It used the old training targets and lost 5 trials.
+ Claude restarted the tuning of all four tasks with the fixed code (log `tune_all3.log`).

## 2. What we measured

**Altered training targets before the fix** (Codex, `data.checks.altered_target_rate`):

| Dataset | Pool affected / gold spans | Rate |
|---|---:|---:|
| CleanCoNLL | 159 / 23,566 | 0.675% |
| BC5CDR | 62 / 9,385 | 0.661% |
| MIT Movie | 25 / 19,238 | 0.130% |
| MASSIVE en-US | 15 / 11,344 | 0.132% |
| MASSIVE fr-FR | 32 / 11,100 | 0.288% |

**Old CleanCoNLL tuning (archived, old targets, dev micro-F1):**

| LoRA targets | Trials | Best dev F1 |
|---|---:|---:|
| task heads only | 7 (4 pruned) | 0.618 |
| encoder + all task heads | 4 (1 pruned) | 0.874 |
| encoder + task head | 6 (4 failed) | 0.871 |

+ In short: when the encoder is adapted, dev F1 is about 25 points higher than with the heads only. The restarted search must confirm this.
+ Trials 15 to 19 failed with `torch.OutOfMemoryError`. The tuning process held 18.8 GiB.

**Labelling speed (Gemma 4 12B, 8 slots):**

+ 190 sentences per minute when the teacher has the GPU alone.
+ About 26 sentences per minute while tuning shares the GPU.
+ CleanCoNLL pool: 6,401 of 13,957 labelled at 15:30.

**Checks:** `just check-cpu`: 135 passed, 50 deselected (model tests). Ruff is clean.

## 3. What failed or stays unknown

+ The 50 GPU model tests did not run, because tuning and the teacher use the GPU.
+ The CrossRE smoke test of `tune` ran out of memory, because the GPU was shared. The Hallmarks and MASSIVE smoke tests passed.
+ Real error, CleanCoNLL trial 19: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 734.00 MiB ... this process has 18.79 GiB memory in use.`
+ Still unknown: whether `free_gpu()` stops the leak on real trials. `gpu_mem.log` records memory for each process every minute.

## 4. What changed on disk

+ `src/active_gliner/`: `exact_targets.py`, `protocol.py` (new), and the files changed in `1ad11ab` and `a98e141`.
+ `tests/test_protocol_fixes.py` (new), `tests/test_relations_pairs.py`.
+ `runs/tuning_archive/cleanconll-old-targets/` (with the old `ner.yaml`) and `runs/tuning_archive/hallmarks-aborted/`.
+ `configs/recipes/` is empty until the new search freezes recipes.

## 5. What the next session must do first

1. Read `gpu_mem.log`. The tuning process memory must stay flat across trials.
2. Check `trials.jsonl` for each task. There must be no `FAIL` rows, and each recipe must be frozen.
3. When the GPU is free, run `just check` with the model tests.
4. Then run the O7 timing gate and the same-seed repeatability check, and start `active-gliner matrix run` with the ground-truth blocks.

## Changelog

- 2026-09-25 15:33 CEST - Created.
