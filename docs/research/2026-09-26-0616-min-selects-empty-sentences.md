---
title: Min selection picks empty sentences on NER (early main-run evidence)
date: 2026-09-26 06:16 CEST
author: Claude
type: research
status: frozen
---

# Min selection picks empty sentences on NER

## 1. Observation

The first `ner_core` runs on CleanCoNLL (ground truth labels, test micro-F1, seeds 1 to 3) show that min selection is far below random:

| Selector | N 100 | N 400 |
|---|---|---|
| random | .820, .808, .821 | .861, .859, .843 |
| diversity | .815, .823, .825 | .870, .858, .866 |
| min | .630, .664, .603 | .712, .665, .727 |
| min, Gemma labels | .607, .614, .639 | .649, .644, .672 |

These are early numbers from `runs/ner/cleanconll/en-US/*/*/metrics.json`. The paper numbers come from the analysis scripts only (research rule 10).

## 2. Mechanism

The frozen confidence rule gives a sentence with no student prediction the confidence 0 (design section 1 and `confidence.min_confidence`). Min selection takes the lowest confidence first, so it takes sentences where the zero-shot student predicts nothing.

Seed 1 selections on CleanCoNLL:

| Selection | Share with no gold entity | Gold entities in the set | Mean words per sentence |
|---|---:|---:|---:|
| min, N 100 | .790 | 30 | 7.4 |
| min, N 400 | .777 | 106 | 6.7 |
| random, N 100 | .220 | 178 | 15.4 |
| random, N 400 | .240 | 640 | 14.6 |
| diversity, N 100 | .240 | 150 | 11.0 |
| diversity, N 400 | .320 | 529 | 10.3 |
| full pool | .208 | | |

The min set gives the student about six times fewer entity examples than a random set of the same size. The pool analysis (`2026-09-25-2328-teacher-vs-uncertainty.md`) showed the same pattern: 74.5% of the least-sure CleanCoNLL decile has no gold entity.

## 3. What we do and do not do

+ We keep the primary rule (`no_prediction: zero`). The design froze it before any result. A switch after seeing results would be selective reporting.
+ The design already contains the sensitivity block `ner_no_prediction_sensitivity` (empty predictions ranked last). It started on 26 Sep at 06:16, ahead of its turn, because it answers the first reviewer question.
+ The paper must report both rules side by side for NER, and explain the mechanism with the table above.

## 4. What it means for the paper

+ The standard min-confidence rule, with "no prediction = lowest confidence", can make uncertainty sampling worse than random on NER, even with perfect labels. The cause is data content, not label noise.
+ The primary interaction (min minus random under Gemma, minus the same gap under ground truth) is still open. It needs the random Gemma arms.

## Changelog

- 2026-09-26 06:16 CEST - Created from the first 24 `ner_core` runs.
