---
title: CrossRE whole-pool runs, out-of-memory failures, and a one-time bad gradient
date: 2026-09-27 11:58 CEST
author: Claude Code
type: update
status: frozen
---

# CrossRE whole-pool runs, out-of-memory failures, and a one-time bad gradient

## 1. What I did

+ Watched the matrix lanes from 09:04 to 11:58.
+ Found two CrossRE runs that ran out of GPU memory, and moved CrossRE out of the `other_tasks` lane.
+ Found the CrossRE ground-truth whole-pool run failed with a non-finite gradient.
+ Replayed that run with a gradient checker to find the cause.

## 2. What I measured

Whole-pool (upper-bound) runs, seed 1, test micro F1. These are report numbers only. The paper takes numbers from the analysis script.

| Run | Test F1 | Best step |
|---|---|---|
| BC5CDR, ground truth | 0.874 | 900 |
| BC5CDR, Gemma 12B | 0.795 | |
| MIT Movie, ground truth | 0.884 | |
| MIT Movie, Gemma 12B | 0.797 | |
| CrossRE, Gemma 12B | 0.125 | 200 |
| CrossRE, ground truth | failed at step 416 | |

Replay of CrossRE ground truth, whole pool (scratch root, not a paper run):

+ No bad gradient in 1000 steps.
+ Gradient norm: median 0.23, largest 0.85 (step 706). No slow growth.
+ Dev F1 by step: 0.003, 0.235, 0.291, 0.347, 0.348, 0.365, 0.409, 0.431 (steps 100 to 800).
+ Best step 800, test F1 0.406.
+ The failed run had dev F1 0.356 at step 400; the replay had 0.347. The two processes took slightly different paths (section 3e of the 26 Sep note).

## 3. What failed or stays unknown

### 3a. Out of memory

The CrossRE whole-pool run holds about 14 GiB. A CrossRE N100 or N200 run needs about 5.5 GiB. Two runs failed in about 11 s each:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 23.79 GiB of which 137.25 MiB is free.
```

+ `relations/crossre/en-US/gemma-4-12b/min-N200-seed3`
+ `relations/crossre/en-US/ground_truth/random-N100-seed1`

Fix: I stopped the `other_tasks` lane and restarted it with `--dataset hallmarks --dataset massive`. Failed runs have no `metrics.json`, so the runner still counts them as pending.

### 3b. Non-finite gradient, CrossRE ground truth, whole pool

```
RuntimeError: The total norm of order 2.0 for gradients from `parameters` is non-finite,
so it cannot be clipped.
```

+ The run failed at step 416 of 1000. The loss at step 415 was 0.078 and finite.
+ The step-416 batch holds 16 normal pairs (72 to 418 characters). No sentence stands out.
+ The gliner2 `SparseRelationScorer` has no norm, log, or unguarded division.
+ The replay did not reproduce it. The cause stays unknown. The best guess is a one-time fault on the aging 3090. No Xid was logged.
+ Handling: rerun the run. No code change, because a "skip bad steps" rule changes the protocol and stales every finished run.
+ If it happens again, the replay script is ready: `tmp/replay_crossre_nan.py` in the job folder. It logs the gradient norm per step and reruns the bad batch under anomaly detection.

### 3c. Gemma CrossRE whole pool collapses

+ Dev F1 was 0.081 at step 100, 0.102 at step 200, then 0.0 from step 300 to 1000.
+ This looks like a teacher effect, not a bug. Gemma relation labels score 0.275 F1 against ground truth on the CrossRE pool. The Gemma N200 runs also land near 0.13 test F1.
+ Open: check whether the model predicts no relation at all after step 300.

### 3d. Offline plan without Cerebras, and a GPU fault with 4 lanes (13:13)

+ Abhishek said to go on without the Cerebras key and without the sudo clock cap. The 21 Cerebras runs (ner_qwen, Qwen and gpt-oss ladder cells) are dropped. The paper says so in Limitations.
+ At 13:09 I started a fourth training lane (E4B ladder). At 13:11 the driver logged `Xid 109 CTX SWITCH TIMEOUT`, and `classification/hallmarks/en-US/ground_truth/random-N100-seed1` died with `CUDA_ERROR_LAUNCH_FAILED`. I stopped the fourth lane. Rule: at most 3 training lanes.
+ A lane tries each run once. Failed runs stay pending. `sweep_after.sh` (job tmp) reruns them after every lane stops.
+ CrossRE ground-truth whole pool, rerun in the chain lane: test F1 0.410, best step 1000. The failure in 3b did not come back.

### 3e. Analysis work while the GPU runs (13:13)

+ The quick table I showed Abhishek at 12:55 was wrong for CleanCoNLL min N400: it mixed the `-nopredlast` folders into min. The right means (true labels): random 0.857, min 0.707, min empty last 0.845. The analysis script had them right.
+ Cost curves (design 3f and the release chart): human labels at 0.50 USD per sentence (Kasner et al. 2026); local teachers at free-GPU time from a new `time-teacher` job; min and diversity pay for pool scoring at the student's zero-shot time per sentence. Cached request times were not used: they were measured while training shared the GPU (E4B median 5.1 s against 1.75 s for the 12B model alone).
+ `analysis/tables.py` writes the main results table and one macro per cell. The Q1 paragraph used macro names the script never wrote; fixed.
+ Learning curves show whole-pool and zero-shot reference lines.
+ Lead (one dataset, 3 seeds): CleanCoNLL with Gemma labels, min empty last N400 = 75.1 against 74.0 for the whole Gemma-labelled pool.
+ Checks: `just check-cpu` 202 passed, 50 deselected (model and teacher tests wait for a free GPU). The paper builds with tectonic.

### 3f. Hallmarks whole pool is low (13:17)

+ `classification/hallmarks/en-US/ground_truth/all-seed1`: test micro-F1 0.423, macro 0.360, best dev 0.379 at step 200, stopped by early stopping at step 500 (about a third of one pass over 12,119 sentences). Dev F1 by check: 0.28, 0.38, 0.30, 0.29, 0.15, while dev loss fell from 0.103 to 0.087.
+ The dev-chosen threshold (0.3) gives dev 0.391, so the threshold is not the cause.
+ Small runs level off at dev 0.35 to 0.41, also when they train all 1000 steps (N100 seeds 2 and 3). All arms sit in one band.
+ Published HoC scores near 0.87 are abstract-level example-based F1, not comparable with sentence-level micro-F1.
+ Decision: no recipe change; the compared arms share one protocol. The paper must say that the Hallmarks whole-pool reference stopped early and is a lower estimate of the ceiling.

### 3g. Evening: GPU faults, out-of-memory CrossRE seeds, one run with no steps (22:38)

+ Xid 109 at 17:18, 20:11, 21:46 and 22:16 with 3 lanes. Each killed one MASSIVE run 3 to 7 minutes in (4 of the last 14 MASSIVE runs). GPU at 63 C and 1800 MHz, so not heat. Kept 3 lanes: the loss is minutes per fault, and the sweep reruns the runs.
+ 20:08: the chain lane reached `other_tasks_primary_extra_seeds`. Its 8 CrossRE runs failed in about 11 s each (out of memory next to a 9.9 GB MASSIVE run). `sweep_after.sh` now runs them in their own lane slot before E4B and the sweep.
+ `slots/massive/en-US/ground_truth/min-N400-seed4` (started 22:18, 2 minutes after a fault): one dev check at step 0, epoch 39, dev loss 335, no `train_log.jsonl`, then `FileNotFoundError` in the training plot. The trainer never counted an optimizer step. No other run has this pattern (all eval logs scanned). Treated as a one-time event; the sweep reruns it. If it fails the same way again, investigate.

## 4. What changed on disk

+ No change in `src/`.
+ Scratch replay under the job folder `tmp/replay_root/`. It is not under `runs/`.
+ This note.

+ Commits 3d3e16a, 8a5b6cb, b1433f4: `timing.py`, `analysis/cost.py`, `analysis/figures.py`, `analysis/tables.py`, `analysis/report.py`, `cli.py`, `justfile`, `configs/prices.yaml`, `paper/main.tex`, `paper/references_extra.bib`, tests `test_cost_curves.py` and `test_tables.py`.
+ Job scripts (not in git): `crossre_after_other.sh`, `sweep_after.sh`, `after_matrix.sh`.

## 5. What the next session must do first

1. Keep 3 training lanes at most. The queue runs itself: CrossRE other_tasks, then E4B, then the failure sweep, then free-GPU timings.
2. When `after_matrix.log` says ALL DONE: check that the zero-shot scores match the archived ones (only timings may change), then run `active-gliner analyse` into `paper/`.
3. Write Results Q1 to Q7 from `numbers.tex` and the tables. Add direction words only after the final numbers.

## Changelog

- 2026-09-27 22:38 CEST - Added section 3g, evening faults and the no-step MASSIVE run.
- 2026-09-27 13:17 CEST - Added section 3f, Hallmarks whole pool.
- 2026-09-27 13:13 CEST - Added sections 3d and 3e, updated sections 4 and 5.

- 2026-09-27 11:58 CEST - Created.
