---
title: Teacher parser fix (word edges) and matrix restart
date: 2026-09-26 07:15 CEST
author: Claude
type: update
status: frozen
---

# Teacher parser fix and matrix restart

## 1. What I did

+ Found why every Gemma-label run failed after about 9 s. The error was `ValueError: Training targets differ from supplied spans: cleanconll:en-US:train-6135`.
+ Cause: `teachers/validate.py` placed a teacher mention at every place its text occurs, also inside other words. Examples:
  + MIT Movie: the rating "r" became spans inside "are", "rated", "directors".
  + CleanCoNLL: "Pakistan" inside the one student word "Pakistan-ruled".
+ Fix (commit `49bb541`): a mention matches only on student word edges (`gliner2` `WhitespaceTokenSplitter`). Text found only inside words is a `not_on_word_edges` validation error.
+ Added 3 tests in `tests/test_teachers.py`. Added design section 17 "Teacher span placement". Added a correction note to `docs/research/2026-09-25-2328-teacher-vs-uncertainty.md`.
+ Stopped all matrix jobs, then restarted three lanes with `--rerun-stale`: `ner_core`, `other_tasks`, and the chain script (`ner_no_prediction_sensitivity` first, then the other runnable blocks).

## 2. What I measured

Cached raw answers re-parsed with the fixed parser (no new teacher calls):

| Pool cache | Spans before | Spans after | Teacher span F1 before | After |
|---|---:|---:|---:|---:|
| CleanCoNLL | 28,934 | 28,788 | .786 | .789 |
| BC5CDR | 19,557 | 19,516 | .793 | .796 |
| MIT Movie | 23,573 | 22,277 | .742 | .776 |
| MASSIVE en-US | 19,824 | 19,784 | .497 | .498 |
| MASSIVE fr-FR | 18,686 | 18,669 | .464 | .464 |

+ Off-edge spans left after the fix: 0 in every cache.
+ Training target check on every Gemma pool record: 51,010 checked, 0 failed.
+ `just check`: lint, format, 227 passed, 1 skipped (live teacher test).

## 3. What failed or stays unknown

+ All 31 finished runs became stale through the code fingerprint and run again (about 10 GPU hours). Old folders move to `runs-archives/`.
+ Reproduction check (07:21): CleanCoNLL ground truth min N100 seed 1 reran with test micro-F1 0.630345853477129, the same to every digit as the archived run (same tp 3390, fp 1641, fn 2335). Best dev F1 differs in the 5th digit (0.6549760 against 0.6549200); same best step 100 and same dev threshold 0.3. The in-training dev evaluation is not bit-exact, likely a non-deterministic kernel that `warn_only=True` lets through. It could matter only in a near-tie of checkpoints. Open.
+ `ner_qwen` still needs the Cerebras key. `teacher_ladder_b` still needs Gemma E4B labels.

## 3b. Strict exact numerics (08:19)

+ Cause of the drift above: memory-efficient attention backward (in the gliner2 heads) is non-deterministic under `warn_only=True`. It was the only warning in a 30-step run.
+ CrossRE training ignored exact numerics (always bf16 autocast).
+ Fix: `train.apply_exact_numerics` (strict mode, no TF32, cuDNN deterministic) for all tasks; CrossRE trains in fp32 under exact numerics. Design section 17 "Strict exact numerics".
+ Test: each task ran twice at the same time in two processes (30 steps, eval every 15, N100, seed 1, ground truth). Losses, dev scores and test micro-F1 are equal to every digit for all four tasks: NER .7415039768618944, slots .49608027594857323, classification .2558746736292428, CrossRE 0.0 (too few steps). No strict-mode error. Peak GPU memory: NER 2.55, slots 7.29, classification 3.73, CrossRE 10.21 GiB.
+ `just check`: 229 passed, 1 skipped.
+ All finished runs are stale again and rerun.

## 3c. First fp32 CrossRE run (08:54)

+ `crossre/ground_truth/min-N100-seed1`: fp32 test .309, best dev .337 at step 300, early stop at step 600 (patience 3). The two archived bf16 attempts ended at dev .402 and .392 (test .351 for the first), after 1000 steps.
+ Dev curves: fp32 .222, .286, .337, .303, .296, .333. bf16 .199, .294, .326, .320, .354, .364, .382, .388, .403, .402 and .200, .223, .319, .340, .359, .366, .350, .392, .389, .392.
+ Reading: fp32 was ahead at step 300; three dips in a row then stopped it while the trend still rose. CrossRE dev F1 is noisy per 100 steps, so patience 3 (design default for all tasks) is fragile on CrossRE.
+ Not changed: patience is a design value. Decide after seeds 2 and 3 (fp32). Options, for a design round: keep 3; raise CrossRE patience; or train CrossRE a fixed 1000 steps and pick the best dev checkpoint.

## 3d. CrossRE seeds 2 and 3, and the empty-last block (10:21)

+ CrossRE ground truth, min, N100, fp32:
  + seed 2: full 1000 steps, best dev .392 at step 900, test .357.
  + seed 3: full 1000 steps, best dev .377 at step 800.
  + seed 1 (above): early stop at step 600, best dev .337, test .309.
+ Reading: fp32 learns as well as bf16. Early stopping cut 1 of 3 runs short. The rule is the same for every arm, so it adds noise, not bias.
+ Recommendation for a design round: CrossRE uses the fixed 1000 steps and keeps the best dev checkpoint (dev only; costs about 40% more time on runs that would stop early). Needs a decision before most CrossRE runs finish; if taken, the finished CrossRE runs rerun.
+ `ner_no_prediction_sensitivity` complete, test micro-F1 seeds 1 to 3: ground truth .848, .848, .838; Gemma .748, .751, .757.

## 3e. CrossRE is not bit-repeatable across processes (12:51, corrected 14:02)

+ The rerun of `crossre/ground_truth/min-N100-seed1` under the new recipe (patience only) should match the 08:19 fp32 attempt up to step 600. Training loss matches for steps 1 to 249 and first differs at step 250 (.06313477 against .06313469). Dev F1 then drifts: step 300 .342 against .337, step 400 .344 against .303.
+ Same selection and data order (losses match for 249 steps). Strict mode raised no error. NER reruns under strict mode matched to every digit over 1000 steps, so the cause is probably in the relation training path, or in GPU sharing (three lanes, 21 of 24 GiB).
+ Effect: noise, not bias; every arm faces it and seeds average it. The claim "same seed, same score" does not hold for CrossRE yet.
+ Next, when the GPU is free: run two 300-step CrossRE runs alone at the same time and in sequence, to separate GPU sharing from a relation-path op.
+ Correction (14:02): `crossre/ground_truth/min-N100-seed2` also reran under the new recipe (its earlier run trained all 1000 steps, so the two should match). Training loss first differs at step 27, not 250. Earlier: test .357, best dev .392 at step 900. Rerun: test .353, best dev .382 at step 1000. The split can start almost at once; the size is about .01 F1 per run. The 30-step paired test (section 3b) matched, so the cause is intermittent.

## 3f. Diversity pairing fix and zero-shot ladder (16:00)

+ The pairing check stopped 3 Gemma diversity runs on CleanCoNLL (N100 seed 3, N400 seeds 1 and 2): "Paired arms selected different pool IDs". Each run computed its own GPU sentence embeddings; rounding moved a sentence to another k-means cluster.
+ Fix (commit `0036756`): embeddings are computed once per pool and code version and saved as `runs/pool_scores/embeddings-*.npy`, like pool scores. Test added. `just check-cpu`: 187 passed.
+ The fix touches protocol code, so every finished run is stale again. All lanes restarted with `--rerun-stale` at 16:00.
+ Zero-shot ladder (lower bound, commit `e44c582`): the first start failed because the pinned `main` snapshots of small, base and large had only a README on disk. Downloaded the pinned snapshots (no upload) and restarted `just zero-shot`.

## 3g. Paper, bibliography, E4B labels, GPU fault (17:40)

+ Analysis dry run (`active-gliner analyse` into a scratch folder): 72 runs read, no crash. Fixed: zero-shot rows (labels `none`) were treated as a teacher (commit `d3e56e1`). Open: the deployment (Pareto) figure needs latency and cost for trained students; the zero-shot latencies were measured on a shared GPU, so clean timings must be taken on a free GPU at the end.
+ Paper setup text updated to the current protocol (commit `57fd95e`). The four "TO CHECK" references were verified against publisher or Crossref records (commit `007c23c`). The paper compiles with no undefined citations.
+ `configs/prices.yaml`: `gemma-4-e4b` replaces the dropped `qwen-3.8-2b` (commit `71ff3e6`).
+ Gemma 4 E4B server on port 8021. Labelled: ladder B selections (random, N400, seeds 1-3): CleanCoNLL 1,161 sentences, valid rate .970; Hallmarks 1,159, valid rate 1.000. Ladder A test samples: CleanCoNLL, BC5CDR, MIT Movie, Hallmarks, MASSIVE en-US.
+ GPU fault 17:24:24: `Xid 109 CTX SWITCH TIMEOUT` killed the CrossRE ladder-sample process while 5 CUDA processes shared the GPU (clock cap 1800 MHz active). Other processes survived. Keep 4 or fewer GPU processes. Lowering the cap to 1700 MHz needs sudo: asked Abhishek. The CrossRE sample reruns after the zero-shot ladder ends (`tmp/e4b_crossre.sh`).
+ Zero-shot ladder: 22 of 28 pairs done at 17:40.

## 3h. Zero-shot ladder complete (18:59)

+ 27 of 28 pairs scored; GLiNER2 large has no native relation scorer, so its CrossRE pair is recorded as `UNSUPPORTED.txt` (commit `017fc61`). GLiNER2 large is an older `SpanExtractor`; slot prediction read `boundary_settings`, fixed in `zero_shot.py` only (commit `557f00a`), so no protocol code changed.
+ Test micro-F1 (small / base / multi / large): CleanCoNLL .511 / .547 / .557 / .545; BC5CDR .596 / .680 / .647 / .701; MIT Movie .522 / .646 / .555 / .584; CrossRE .025 / .072 / .084 / n/a; Hallmarks .186 / .201 / .132 / .203; MASSIVE en-US .324 / .398 / .316 / .152; MASSIVE fr-FR .226 / .258 / .262 / .110.
+ Latencies in these runs were measured on a shared GPU; time them again on a free GPU before the deployment figure.
+ E4B labels complete, including the CrossRE ladder sample (1,000 sentences, 0 retries). E4B server stopped. CrossRE matrix lane restarted at 18:59.

## 3i. Diversity k-means made single-thread (23:33)

+ After the shared-embedding fix, 4 CleanCoNLL Gemma diversity runs still failed the pairing check (N100 seed 3, N400 seeds 1 to 3).
+ Test on the shared embedding file: 3 separate processes gave 2 different selections (`c644…`, `d895…`, `c644…`). Multi-threaded k-means sums in a varying order.
+ With one thread (`threadpool_limits(1)`): 4 of 4 identical, and 3 of 3 with the committed code. N400 selection takes about 12 s.
+ The selection key now includes the code fingerprint, so an old selection cannot match a new one.
+ `threadpoolctl==3.7.0` declared (already installed with scikit-learn). Tests: cross-process identity and one-thread limit. `just check`: 240 passed, 1 skipped (commit `5a5a4d6`).
+ All finished runs are stale again (60 done before the fix). Lanes restarted with `--rerun-stale` at 23:33.

## 4. What changed on disk

+ `src/active_gliner/teachers/validate.py`
+ `tests/test_teachers.py`
+ `docs/research/2026-09-25-1113-full-merge-design.md` (section 17, changelog)
+ `docs/research/2026-09-25-2328-teacher-vs-uncertainty.md` (changelog correction)

## 5. What the next session must do first

1. Check `runs/matrix_log.jsonl` for failed rows and read any `FAILED.txt`.
2. Compare one rerun ground-truth score with its archived copy in `runs-archives/`.
3. Recompute the teacher-versus-uncertainty tables through the analysis script on the fixed parse.

## Changelog

- 2026-09-26 23:33 CEST - Added section 3i, single-thread k-means.
- 2026-09-26 18:59 CEST - Added section 3h, zero-shot ladder complete.
- 2026-09-26 17:40 CEST - Added section 3g, paper, bibliography, E4B labels, GPU fault.
- 2026-09-26 16:00 CEST - Added section 3f, diversity pairing fix and zero-shot ladder.
- 2026-09-26 14:02 CEST - Section 3e corrected: CrossRE seed 2 splits at step 27.
- 2026-09-26 12:51 CEST - Added section 3e, CrossRE drift after step 250.
- 2026-09-26 12:20 CEST - CrossRE decision taken: fixed 1000 steps (design section 17).
- 2026-09-26 10:21 CEST - Added section 3d, CrossRE seeds 2 and 3, empty-last block results.
- 2026-09-26 08:54 CEST - Added section 3c, first fp32 CrossRE run and early-stopping note.
- 2026-09-26 08:19 CEST - Added section 3b, strict exact numerics.
- 2026-09-26 07:21 CEST - Added the reproduction check.
- 2026-09-26 07:15 CEST - Created.
