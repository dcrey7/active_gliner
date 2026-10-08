---
title: Mixing curves on every dataset, in the paper
date: 2026-10-08 11:40 CEST
author: Claude
type: update
status: done
---

# Mixing curves on every dataset, in the paper

## 1. What was done

+ Curves - thesis Fig 4.6.1 on all seven dataset-locale places: ranked selection (min, empty last), human share 0/25/50/75/100% at N = 100, 400, 1000, 2500 and the whole pool; random lines at 0% and 100%.
+ Minimum curve - whole pool with H nested human labels and Gemma on the rest, H = 25 to 2500.
+ Dev labels - Gemma 4 12B labelled every dev split (72 minutes, 8 parallel slots).
+ Analysis - `curves.py` (figure, table `tables/curves.tex`, macros), `minimum.py` (dev choice, one test check, Holm), `reliability.py` (confidence against share correct).
+ Paper - Section 4.1 "How many human labels does it take to beat the LLM?" replaces the N = 400 mixing table; abstract, contributions and thesis section updated.
+ Two-step training (teacher labels then human labels) - built and committed (b7d6ce9), then taken off the queue: Abhishek's method ranks sentences and labels the top N, which the curves already test. No two-step run exists.

## 2. What was measured

+ Smallest human share that beats Gemma falls with N. From N = 1000: 25% or less on six of seven places; 0% on MIT Movie and MASSIVE; Hallmarks needs 75%.
+ Dev-chosen fewest human labels (human + Gemma sentences), test gap over Gemma, Holm p:

| Place | Cell | Test F1 | Gap | p |
|---|---|---|---|---|
| MIT Movie | 0+100 | 76.4 | 2.5 | <0.001 |
| MASSIVE en | 0+1000 | 51.7 | 3.0 | <0.001 |
| MASSIVE fr | 0+400 | 47.2 | 1.7 | 0.052 |
| CleanCoNLL | 50+50 | 77.1 | 1.0 | 0.164 |
| CrossRE | 75+25 | 30.4 | 7.7 | <0.001 |
| BC5CDR | 250+750 | 79.9 | 0.9 | 0.042 |
| Hallmarks | 750+250 | 46.9 | 4.0 | <0.001 |

+ Ranked minus random, same N: mean +0.4 points with Gemma labels and with human labels, range -7.5 to +4.0.
+ Gemma labels on the whole rest of the pool help only with about 50 human labels (200 on Hallmarks) and hurt beyond, by up to 18.6 points (MASSIVE en, 1000 human).

## 3. What failed or stays unknown

+ Two CrossRE whole-pool runs (seed 2: 75% share, and H = 25) stopped with a non-finite gradient; rerun started 2026-10-08 10:28.
+ 21 API-teacher runs (Qwen, gpt-oss) never ran; the paper says so.
+ Hallmarks whole-pool human reference has seed SD 8.2.

## 4. What changed on disk

+ `src/active_gliner/analysis/{curves,minimum,reliability,report,aggregate}.py`, `src/active_gliner/{run,matrix}.py`
+ `tests/test_{curves,minimum,matrix}.py`
+ `paper/main.tex`, `paper/numbers.tex`, `paper/tables/curves.tex`, `paper/figures/{mixing-curves,minimum-human-labels,reliability}.*`
+ `docs/research/2026-10-06-1529-two-step-training-design.md`

## 5. Next

1. When the CrossRE reruns end: `active-gliner analyse`, rebuild, check for `??`.
2. Push through `public_squash.sh`, open the PR, merge.

## Changelog

- 2026-10-08 11:40 CEST - Created.
