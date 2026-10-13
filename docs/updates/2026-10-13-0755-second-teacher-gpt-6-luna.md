---
title: Second teacher family, GPT-6 Luna, in the paper
date: 2026-10-13 07:55 CEST
author: Claude
type: update
status: done
---

# Second teacher family, GPT-6 Luna, in the paper

## 1. What was done

+ Teacher - OpenAI `gpt-6-luna` replaces the Cerebras API teachers that never ran.
  Config: `configs/teachers/gpt-6-luna.yaml`. Same frozen prompt and JSON schema as
  Gemma, temperature 0, `reasoning_effort: none`, `max_completion_tokens`.
+ Labels - every test split, plus the selected sentences of 42 runs
  (`teacher_family` block: min and random, main budget, seeds 1 to 3, 7 places).
+ Analysis - new module `src/active_gliner/analysis/second_teacher.py`. It writes
  `paper/tables/second_teacher.tex` and the Luna macros. Tests in
  `tests/test_second_teacher.py`.
+ Paper - Teachers setup, new paragraph "A second teacher family" with Table 3,
  Limitations, and the teacher audit paragraph.

## 2. What was measured

Test micro-F1 of the teacher labels (full test splits):

| Dataset | Gemma 4 12B | GPT-6 Luna | Error overlap (Jaccard) |
|---|---|---|---|
| CleanCoNLL | 76.1 | 79.4 | 0.69 |
| BC5CDR | 79.0 | 79.0 | 0.58 |
| MIT Movie | 74.0 | 68.8 | 0.64 |
| CrossRE | 22.6 | 33.3 | 0.81 |
| Hallmarks | 42.9 | 42.0 | 0.62 |
| MASSIVE en | 48.7 | 55.9 | 0.65 |
| MASSIVE fr | 45.6 | 48.7 | 0.65 |

+ Teacher - Luna better on 4 of 7, mean +2.61 points.
+ Student - min-selected students on Luna labels are +0.26 points above Gemma on average.
+ Interaction - (min - random) teacher minus (min - random) ground truth, 3 shared seeds:
  Luna 0.68, Gemma 0.88. Descriptive, outside the Holm families.
+ Cost - test labelling 2.69 USD (from `label` logs). Valid rate 97.7 to 100 percent.
+ Wall time - 42 runs on 3 GPU lanes, 00:25 to 07:43. MASSIVE and CrossRE about 4 h each.
+ Astra - `gpt-6-astra` costs 100 times Luna (10 / 50 USD per 1M tokens). Not used.

## 3. What failed or stays unknown

+ Luna has no whole-pool labels, so "Teacher bins" for Luna are skipped in `analyse`.
  This is expected.
+ Luna covers only the main-table cells with 3 seeds. The mixing curves and the
  Holm-tested contrasts use Gemma alone.
+ A decision-model dataset (VAST or FinEntity from the Jev Decision Index suite) is
  deferred to v2.

## 4. What changed on disk

+ `src/active_gliner/analysis/second_teacher.py` (new), `report.py`
+ `tests/test_second_teacher.py` (new)
+ `paper/main.tex`, `paper/numbers.tex`, `paper/tables/second_teacher.tex`, `paper/results.json`, `paper/main.pdf`

## 5. What the next session must do first

+ Publish through the job `public_squash.sh`, open the PR, merge.
+ Prepare the arXiv package. The owner submits with his own account.

## Changelog

- 2026-10-13 07:55 CEST - Created.
