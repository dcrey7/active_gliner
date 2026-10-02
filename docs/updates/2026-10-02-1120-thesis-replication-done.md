---
title: Thesis replication finished and written into the paper
date: 2026-10-02 11:20 CEST
author: Claude (for Abhishek Thomas)
type: update
status: done
---

# Thesis replication finished and written into the paper

## 1. What I did

+ Runs - all 405 thesis replication runs finished (selectors, budgets, mixing, LoRA layers, full fine-tune). Matrix: 693 of 714 done, 0 failed, 0 stale. The 21 pending runs are the dropped Qwen and Cerebras teacher cells.
+ Reruns - 6 failed CrossRE runs (4 non-finite gradient, 2 GPU memory) and the Hallmarks full fine-tune ran again with the same config and seed and finished normally. No code change.
+ OOM audit - no study run skipped a batch. The only flagged logs are old trials in `runs/tuning_archive/`. CrossRE logs have no epoch field; its training loop never swallows OOM.
+ New analysis module `src/active_gliner/analysis/selector_audit.py`, with tests in `tests/test_selector_audit.py`.
+ `FinishedRuns` now counts variant runs too (720 = 651 trained main runs + 27 zero-shot + 42 variants).
+ Paper - new section 4.1 "Replicating the thesis experiments", mixing table on all datasets, abstract sentence, thesis section sentence, Limitations fixes.

## 2. What I measured

+ avg, MSE and MNLP select the same sentences in 63 of 63 cells. For NER, min selects that set too (27 of 27).
+ Reason: every pool has more no-prediction sentences than the largest thesis budget, from 563 (6.4%, MIT Movie) to 10,545 (87.0%, Hallmarks). All three scores rank those first.
+ Same sentences and seeds, different F1 = training noise: up to 4.02 points (Hallmarks, ground truth), 0.00 to 0.01 for NER with ground truth, 0.68 for CleanCoNLL with Gemma labels.
+ Mixing (min, main budget): 50% ground truth matches 100% only on MIT Movie (78.16 against 78.53). Elsewhere each share helps step by step (MASSIVE en 48.83, 53.73, 58.43, 64.21, 69.71); on CleanCoNLL, BC5CDR and Hallmarks the first 25% gives almost nothing.
+ Variants (random, N 400, ground truth, 3 seeds): encoder-only LoRA within 0.7 points of the default everywhere; heads-only loses 9 to 21 points, 4.13 on Hallmarks.
+ Full fine-tune (whole pool, 1 seed, encoder lr 1e-5 not tuned) is below LoRA whole pool on 4 of 5 datasets; above only on Hallmarks (51.41 against 42.32, where LoRA stopped early).
+ Threshold spread on dev: zero-shot up to 11.92 points, trained at most 1.99.
+ Repaired runs moved the primary NER interaction from 3.37 to 3.32 and the classification p from 0.336 to 0.512. No conclusion changed.
+ `just check`: 291 passed, 1 skipped. Paper builds with tectonic: 11 pages, no warnings, no "??".

## 3. What failed or stays unknown

+ The earlier claim that only NER is bit-repeatable is too strong: NER on Gemma labels also varies (0.28 to 0.68 points). The paper now says this.
+ Full fine-tune has one seed and an untuned learning rate.
+ Synthetic data (thesis E12) not run.

## 4. What changed on disk

+ `src/active_gliner/analysis/selector_audit.py` (new), `src/active_gliner/analysis/report.py`
+ `tests/test_selector_audit.py` (new)
+ `paper/main.tex`, `paper/numbers.tex`, `paper/results.json`, `paper/tables/`, `paper/figures/`, `paper/main.pdf`
+ Job scripts: `selector_overlap.py`, `no_prediction_share.py`, `crossre_fail_steps.py` in the job `tmp/`.

## 5. What the next session does first

1. Push through the clean public squash branch and merge the PR.
2. Optional: thesis E12 synthetic data (needs the Gemma server).

## Changelog

- 2026-10-02 11:20 CEST - Created.
