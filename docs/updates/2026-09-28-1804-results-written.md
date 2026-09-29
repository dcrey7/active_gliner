---
title: Offline matrix done, analysis and Results section written
date: 2026-09-28 18:04 CEST
author: Claude (for Abhishek Thomas)
type: update
status: frozen
---

# Offline matrix done, analysis and Results section written

## 1. What I did

+ Matrix - all offline runs finished: 288 of 309 planned cells, 315 finished runs with sensitivity and extra seeds. The 21 Cerebras cells (Qwen, gpt-oss) are dropped, by Abhishek's word. The paper says so in Limitations.
+ GPU faults - the matrix child retries a run in a fresh process (`os.execve`) after out-of-memory, CUBLAS alloc failure or an Xid 109 launch failure. Up to 10 tries, 120 s apart.
+ Timings - teacher and zero-shot timings on a free GPU (`time-teacher`, `runs/timing/`).
+ Cost - labelling cost per run (`analysis/cost.py`), with ground truth at a simulated 0.50 USD per sentence (Kasner et al. 2026).
+ Deployment figures - pooled NER points, Pareto and quadrant charts (`analysis/deployment.py`).
+ Teacher ladder - Gemma 4 12B against E4B on the frozen test samples (`analysis/ladder.py: scores`). Each sentence weighs 1 / inclusion probability. The gap has a bootstrap interval that resamples whole documents within strata.
+ Macros - every Results number is a macro in `paper/numbers.tex` (rule 10): contrasts, cells, teacher F1 and bins, ladder, deployment.
+ Paper - Results Q1 to Q7, abstract result, limitations, appendix with Pareto, learning curves and cost curves. Clean tectonic build, 10 pages, no `??`.

## 2. What I measured

Interaction Δ = (min - random) with Gemma - (min - random) with ground truth, F1 points, 95% test bootstrap, Holm:

| Task | Δ | interval | p | practical (min - random, Gemma) |
|---|---|---|---|---|
| NER | +3.37 | 2.94 to 3.85 | <0.001 | -6.92 |
| Relations | -3.23 | -4.80 to -1.91 | <0.001 | +1.37 |
| Classification | -1.29 | -3.14 to 0.52 | 0.336 | -3.64 |
| Slots | +0.64 | -0.47 to 1.78 | 0.336 | +0.36 (p 0.390) |

+ CleanCoNLL - Gemma min 65.2, min with empty sentences last 75.1, random 72.2, whole pool 74.0.
+ Q2 - NER teacher F1 rises with student confidence (BC5CDR 70.3 to 88.0). CrossRE, Hallmarks and MASSIVE fall at the most-sure end (CrossRE 29.4 to 12.7).
+ Ladder - 12B beats E4B by 4.3 to 13.6 points on 5 of 6 datasets. CrossRE gap 0.6 (-1.9 to 3.0). Error Jaccard 0.60 to 0.70, CrossRE 0.90.
+ Students - E4B labels give weaker students (CleanCoNLL 68.2 against 72.2).
+ Beating the teacher - whole-pool Gemma students beat Gemma on MIT Movie, BC5CDR and MASSIVE en.
+ Deployment NER - Gemma 76.4 F1 at 329 ms per sentence. The student on 400 random Gemma labels reaches 75.3 at 2.0 ms, 165 times faster, 20.11 against 0.12 USD per 1M sentences.
+ Tests - `just check` 274 passed, 1 skipped.

## 3. What failed or stays unknown

+ CleanCoNLL ladder sample - 52 documents. Its level estimate for 12B is 65.2 against 76.1 on the full test. The paired gap is fine. The paper says so.
+ Hallmarks whole pool - stopped early on a dev plateau (0.423). It understates the ceiling. Noted in Limitations.
+ Fine-tuned student latency - not measured. It reuses the base model timing. Noted in Limitations.
+ Cross-family teacher overlap - unknown, because no Cerebras teachers ran.
+ Figure fonts - small at column width. Still readable.

## 4. What changed on disk

Branch `worktree-v2-research-plan`, local only, not pushed. Commits since 913d9d1:

+ `2b19753` - deployment figure labels and y-axis.
+ `a95b647` - first paper outputs.
+ `4c26607` - ladder scores, new macros, minus signs, p format.
+ `6967d78` - `data/ladder/` frozen samples.
+ `272a8b2` - paper Results, abstract, limitations, appendix; `paper/main.pdf`.

Files: `src/active_gliner/analysis/{ladder,tables,report,deployment,figures,numbers,teacher}.py`, `tests/test_{ladder,tables,deployment,analysis}.py`, `paper/main.tex`, `paper/numbers.tex`, `paper/results.json`, `paper/tables/`, `paper/figures/`.

## 5. What the next session does first

1. Read `paper/main.pdf` end to end. Check each Results claim against Table 2 and `numbers.tex`.
2. Decide on the Cerebras teachers. If a key comes, run the 21 cells and rerun `active-gliner analyse --runs runs --out paper`.
3. Polish figures (larger fonts at column width) and write the supplementary contamination audit.
4. Ask Abhishek before any push, arXiv upload or PyPI release.

## Changelog

- 2026-09-28 18:04 CEST - Created.
- 2026-09-28 18:05 CEST - The contamination audit is now paper Appendix B (commit fcbd179). Step 3 of section 5 no longer needs a supplement, only the figure polish.
