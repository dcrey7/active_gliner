---
title: Mixing curves with GPT-6 Luna as the teacher
date: 2026-10-16 20:39 CEST
author: Claude
type: update
status: done
---

# Mixing curves with GPT-6 Luna as the teacher

## 1. What was done

+ Runs - new matrix block `teacher_family_curve`: ranked min (empty last), human shares
  0/25/50/75 at N 100/400/1000/2500 and the whole pool, 3 seeds, 7 places = 408 runs.
  Human-only and random cells are shared with the Gemma curves.
+ Labels - Luna labelled every pool and dev split (dev is needed for the dev-chosen check).
+ Analysis - `curves.curve_rows`, `figure`, `macros` and `fewest_macros` take a teacher
  or tag. New `curves.summary_counts`. Report writes `paper/tables/curves_luna.tex`,
  `paper/figures/mixing-curves-luna.*`, Luna macros and `fewest_ranked_luna` in results.
+ Paper - Table 5, Figure 9 (appendix), paragraph "Against a second LLM family",
  abstract sentence, Teachers setup and Limitations updated.

## 2. What was measured

+ Quarter rule - from N = 1,000, a human share of 25 percent or less beats Luna on 6 of 7
  (Gemma: 6 of 7). Hallmarks needs 50 percent against Luna.
+ Dev-chosen fewest human labels - confirmed on test on 7 of 7 for Luna (Gemma: 5 of 7).

| Dataset | Gemma fewest (human+LLM) | Luna fewest | Luna test gap, Holm p |
|---|---|---|---|
| MIT Movie | 0+100 | 0+400 | +3.75, <0.001 |
| MASSIVE en | 0+1000 | 50+50 | +2.01, 0.032 |
| MASSIVE fr | 0+400 (not confirmed) | 75+25 | +2.54, 0.020 |
| CleanCoNLL | 50+50 (not confirmed) | 100+0 | +1.82, 0.020 |
| BC5CDR | 250+750 | 200+200 | +0.91, 0.032 |
| Hallmarks | 750+250 | 750+250 | +3.00, 0.024 |
| CrossRE | 75+25 | 1000+0 | +9.35, <0.001 |

+ Reading - a stronger teacher needs more human labels; a weaker one is beaten with none.
+ OpenAI cost - test 2.69 + pools 7.45 + dev 1.72 = 11.86 USD (from `label` logs).
+ Wall time - 13 Oct 18:07 to 16 Oct 19:19, plus a 2-run CrossRE rerun.

## 3. What failed or stays unknown

+ Freeze - on 15 Oct two parallel trainings filled the GPU (24.1 of 24.6 GB) and froze
  the desktop. Fixed: one training at a time, under 20 GB. One Hallmarks run was killed
  and rerun.
+ CrossRE - 2 runs stopped once with "The total norm of order 2.0 for gradients from
  `parameters` is non-finite". A plain rerun finished both (known CrossRE issue).
+ Not run with Luna - the minimum-human-labels curve and the Holm interaction contrasts.

## 4. What changed on disk

+ `src/active_gliner/matrix.py`, `analysis/curves.py`, `analysis/report.py`
+ `tests/test_matrix.py`, `tests/test_curves.py`
+ `paper/main.tex`, `paper/numbers.tex`, `paper/tables/curves_luna.tex`,
  `paper/figures/mixing-curves-luna.*`, `paper/results.json`, `paper/main.pdf`

## 5. What the next session must do first

+ Prepare the arXiv package. The owner submits with his own account.

## Changelog

- 2026-10-16 20:39 CEST - Created.
