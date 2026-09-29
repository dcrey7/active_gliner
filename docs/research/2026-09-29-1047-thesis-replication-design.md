---
title: Thesis replication on all datasets - design addendum
date: 2026-09-29 10:47 CEST
author: Claude (for Abhishek Thomas)
type: research
status: frozen
---

# Thesis replication on all datasets - design addendum

Abhishek asked on 29 Sep 2026 to replicate every thesis experiment with the new code on all datasets and put the results in the paper. This note adds to the frozen design (`2026-09-25-1113-full-merge-design.md`). It changes no existing arm.

## 1. Mapping

| Thesis | Replication | Block |
|---|---|---|
| E7 selectors (min, avg, MSE, MNLP, random) | avg, MSE, MNLP added; thesis formulas, thesis rules for sentences with no prediction | `thesis_selectors` |
| E7 budgets (14 budgets, 10 to 2,500) | min and random at N 50, 200, 1000 (CrossRE 50, 400, 1000) next to the existing 100 and 400 | `thesis_budgets` |
| E10 mixing grid | 25, 50, 75% ground truth, random share, min, Gemma, main budget, on every dataset | `thesis_mixing` |
| E4 LoRA layer study | encoder only, task heads only, against the frozen encoder + heads recipe | `thesis_lora_layers` |
| E2 full fine-tune | whole pool, ground truth, no LoRA | `thesis_full_finetune` |
| E6 threshold sweep | test F1 at thresholds 0.1 to 0.9 from saved models, descriptive | analysis |
| E12 synthetic data | NER only, as in the thesis; Gemma writes sentences with entities | later, own note |
| E1, E5, E8, E9, E11, E13, E14 | already in the study | - |

## 2. Selectors (thesis `selection/strategy.py`)

For each pool sentence, take the confidences of the predicted items: NER and slot spans, relation probabilities, the probabilities of predicted classification labels.

+ avg - mean confidence; no prediction gives 0.
+ MSE - mean of (1 - c)^2; no prediction gives 1.
+ MNLP - mean of -log(c); no prediction gives infinity.
+ Least sure first: lowest avg, highest MSE, highest MNLP. Ties break by the seed, as for min.

The item confidences go to a second pool file. The existing pool score file stays byte-identical, so old selections are unchanged.

## 3. Arms

All blocks use 3 seeds and exact numerics.

+ `thesis_selectors` - avg, MSE, MNLP; ground truth at N 100 and the main budget; Gemma at the main budget. 7 dataset-locales: 189 runs.
+ `thesis_budgets` - min and random, ground truth: 126 runs.
+ `thesis_mixing` - the missing fractions: 48 runs.
+ `thesis_lora_layers` - random, ground truth, main budget; not CrossRE, whose frozen recipe fixes its targets: 36 runs.
+ `thesis_full_finetune` - one seed per dataset-locale, not CrossRE (its custom relation head is not part of a full-model save): 6 runs.

405 runs. At the measured median times and 3 lanes, about 50 GPU hours.

## 4. Keeping the 315 finished runs

The new selectors change protocol code, so the code fingerprint changes. The old runs stay valid only if the new code gives the same numbers.

1. Recorded before the change: code fingerprint `74a9108a7f33d43be0c497f4e3465625d1fcae540bf14cfbb41988696a93a473`.
2. New config fields enter the fingerprint only when set away from their default, so old configs hash as before.
3. Check - the new code rebuilds each pool score file byte-identical to the old one, and reruns of one finished run per task give the same test F1 to every digit (CrossRE: same selection; F1 may differ slightly, as documented).
4. Only then does `configs/protocol_equivalence.yaml` list the old fingerprint with the check result. The matrix treats a run as current when it matches the current code or a listed, checked fingerprint.

## 5. Analysis

+ Selector table - avg, MSE and MNLP against min and random per dataset. Contrast: each selector minus random, paired by seed. Descriptive, outside the Holm families.
+ Learning curves at five budgets.
+ Mixing curve per dataset.
+ LoRA layer and full fine-tune table.

## Changelog

- 2026-09-29 10:47 CEST - Created.
- 2026-09-29 10:54 CEST - Full fine-tune without CrossRE (6 runs, 405 in total). Variant runs get their own folder suffix and stay out of main-run analysis. The thesis docstring examples for MSE and MNLP are miscomputed; the formulas are used as written.
