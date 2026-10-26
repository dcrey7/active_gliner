---
title: Decision tasks, UNER EWT and the cross-task tolerance answer
date: 2026-10-26 01:13 CET
author: Abhishek Thomas
type: update
status: done
---

# Decision tasks, UNER EWT and the cross-task tolerance answer

## 1. What was done

+ Datasets - BoolQ and CLINC150 (decision tasks) and UNER EWT (open NER) added; SST-5
  loaded and labelled but held back (licence unknown, binned crowd sliders). Open
  licence check of every dataset; CleanCoNLL and MIT Movie stay with honest licence notes.
+ Runs - 432 new runs on one GPU lane: decision_curve 288, open_ner_curve 144. Finished
  24 Oct 18:46; zero failures after the BoolQ fix.
+ Labels - Gemma 4 12B and GPT-6 Luna on test, dev and pool of every new set. Luna cost
  about 6 USD in total for the new sets.
+ Analysis - decision and UNER families with their own Holm tests; tolerance table with
  points lost at 25/50/75/100% LLM labels and a 2-point safe share; teacher error table.
+ Paper - new subsection "Which tasks tolerate LLM labels?", Conclusion paragraph, abstract
  passage, Limitations, Ethics (licences), audit paragraph, appendix figures.
+ Review - Codex read-only review of code and protocol (1 major, 3 minor, all fixed or
  reworded); a second Codex review of the paper text.

## 2. What was measured

+ Points lost at 1,000 ranked inputs, 25% LLM labels (Gemma): NER 0.5 to 1.5, CrossRE 2.8,
  Hallmarks 1.7, MASSIVE en 6.1, fr 5.1. Decisions at 50% LLM labels: at most 0.8.
+ No human share beats the teacher on decisions (CLINC150 teacher 86.25 / 90.92 vs best
  student 84.13; BoolQ teacher 90.12 / 77.34 vs best student 70.83, majority 62.17).
+ MASSIVE en: Gemma fills slots in 70% of utterances with no slot; CleanCoNLL: 79% of
  same-span type errors are organisations labelled as locations (sports teams).
+ BoolQ batch 8 vs 4 x 2 on 12 shared cells: mean -0.50, max 1.83 points.

## 3. What failed or stays unknown

+ An 18 GiB GPU cap crashed 15 of 27 BoolQ runs (long passages) and 10 OOM retries wasted
  about 9 hours. Only CLINC150 had been smoke-tested. Fix: BoolQ batch 4 x grad_accum 2,
  cap 21 GiB (owner allows about 23 GB), 2 retries. The old runs are in runs/held.
+ BoolQ 4 x 2 is not identical to batch 8 (length grouping, last partial batch); stated
  in the paper.
+ A leak of runs/held into the analysis was caught before it reached the paper.
+ No prompt gate for UNER EWT (inherits the CleanCoNLL prompt); decision recipe not tuned.

## 3b. Second Codex review of the paper (applied 01:31)

+ Accepted - convention text now descriptive (prompts hold short definitions, ORG names
  teams); exposure claim softened; BC5CDR PubMed copyright note (labels released as ids
  and offsets); Gemma and MASSIVE en named in the abstract; Luna caveat on the heuristic;
  "Both" column in the teacher error table; scope "nine datasets (ten with both MASSIVE
  languages)"; BoolQ length and UNER counts from macros; UNER Luna dev cell vs test grid.
+ Challenged - "no test of where hard examples go": the ner_routing block tested it for
  NER (hardest quarter to humans 62.86 / 59.49 / 75.70 vs random quarter 65.18 / 64.87 /
  75.03). Now cited in the Conclusion, with the empty-prediction caveat.
+ Macro diff - only 14 new macros, no old value changed. Paper 19 pages, 0 "??".
  `just check`: 332 passed, 1 skipped.

## 3c. Story and plain-language pass (the author only, no Codex)

+ Abstract 534 -> 278 words; the introduction says why LLM labels are used (humans slow
  and costly, an LLM on every input slow, costly, sends data away; student 165x faster).
+ The introduction asks three plain questions and answers each in one sentence; new
  overview diagram (Figure 1); four contributions instead of six.
+ "Words we use" paragraph in Setup (student, teacher, pool, dev, test, ground truth,
  zero-shot, cell, seed, micro-F1, LoRA); jargon replaced (QAT/GGUF, TF32, "recipe").
+ Results follow the three questions; Q1-Q7 labels replaced by plain titles.
+ Thesis material in one short section; training tips moved to Appendix B.
+ Conclusion gains a four-step recipe.
+ Removed: Pareto, learning-curve and cost figures (unreadable or repeated); false
  claim that ASO is reported; internal "rule 4" reference; stale median-line caption.
+ Figures redrawn near print size (readable text); teacher chart shows Gemma on the
  seven main datasets only; decision curves enlarged; teacher error table full width.
+ CLINC150 intents and out-of-scope scored apart (student beats Gemma on intents,
  finds 52% of out-of-scope; 250 in pool vs 1,000 in test).
+ All 26 arXiv citations checked against arXiv (title and first author match).
+ Checks: macros unchanged, 18 pages, no "??", `just check` 333 passed.

## 3d. The tool, the README, the notebooks, no AI author

+ The tool learns the decision task (`label` per row) and the paper's recipe:
  `active-gliner rank` (zero-shot ranking, empty predictions last, a random share to
  humans), then `fit --select all --human <file>`.
+ `fit`, `rank` and `predict` use the paper's tuned settings per task type
  (`src/active_gliner/recipes/`, a tested copy of `configs/recipes/`). Relations load the
  native relation head, as the paper runs do.
+ Bug found by the notebooks: before this, `fit` used generic settings. Relations trained
  to 0.0 dev F1 (no relation head), and the decision run ran out of GPU memory
  (augmentation on, LoRA on every task head).
+ One notebook per task type (`notebooks/task_*.ipynb`, built by
  `notebooks/build_task_notebooks.py`, run by `just task-notebooks`).
+ `fit` trains 1,000 steps and checks dev every 100 steps by default, as the paper does
  (before: 500 and 50).
+ Notebook results, dev F1 zero-shot -> trained, one run, seed 1, 1,000 ranked texts,
  250 human, Gemma 4 12B for the rest: NER 47.4 -> 74.6, relations 9.6 -> 21.4,
  multi-label 21.0 -> 43.6, slots 25.4 -> 63.6, decision 68.7 -> 92.7. Relations sits
  between the paper's CrossRE points at 1,000 texts (20.5 LLM only, 27.1 with 25%
  ground truth; test, mean over seeds).
+ `just check`: 339 passed, 1 skipped (GPU tests included).
+ README rewritten: five task types, the recipe, file formats, per-task tolerance.
+ Docs and AGENTS.md name the author, not an AI product; the paper's AI disclosure names
  no product.

## 4. What changed on disk

+ src/active_gliner/data/uner_ewt.py, scripts/fetch_uner_ewt.py, data/splits/uner_ewt-en-US.json
+ src/active_gliner/matrix.py (blocks, MEMORY_SPLIT), matrix_run.py (GPU cap, retries)
+ src/active_gliner/analysis/{report,tolerance,teacher_errors,aggregate,curves,tables}.py
+ paper/main.tex, numbers.tex, results.json, tables/tolerance.tex, tables/teacher_errors.tex
+ tests/test_uner_ewt.py, test_teacher_errors.py, test_tolerance.py, test_matrix.py

## 5. What the next session must do first

+ Done 01:40 - PR 10 (this work) and PR 11 (`just arxiv`) merged into master.
+ The owner submits dist/arxiv.tar.gz to arXiv from his account (copy in ~/Downloads/work).
+ The owner approved the BC5CDR rule: labels released as ids and offsets, never the text.

## Changelog

- 2026-10-26 01:13 CET - Created.
- 2026-10-26 01:31 CET - Second Codex review applied.
- 2026-10-26 01:45 CET - Published (PR 10, 11); arXiv package built.
- 2026-10-26 12:35 CET - Story and plain-language pass; CLINC150 out-of-scope split.
- 2026-10-26 15:52 CET - Tool recipes, rank and human mixing; README; task notebooks.
