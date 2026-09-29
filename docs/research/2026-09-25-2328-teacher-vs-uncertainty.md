---
title: Is the teacher worse where the student is unsure? (descriptive, pool only)
date: 2026-09-25 23:28 CEST
author: Claude
type: research
status: frozen
---

# Is the teacher worse where the student is unsure?

## 1. Question

Does Gemma 4 12B (the teacher) make more errors on the pool sentences that the zero-shot student (GLiNER2.5-multi) is least sure about?

This is descriptive. It uses the training pool only, never dev or test. It changes no design choice.

## 2. Method

+ Script: `2026-09-25-2328-teacher-vs-uncertainty.py` (this folder). Raw output: `2026-09-25-2328-teacher-vs-uncertainty.jsonl`.
+ The zero-shot student scores the full pool at the acquisition threshold 0.5, with the frozen confidence rules:
  + NER and slots: minimum span confidence. A sentence with no prediction gets confidence 0.
  + Classification: min |p - 0.5|.
+ The pool is sorted by confidence and cut into 10 equal groups (decile 1 = least sure). We also draw a seeded random 10%.
+ Teacher labels come from the on-disk cache (prompt versions as frozen in `configs/prompts.yaml`).
+ Metric: pooled micro-F1 of the teacher labels against ground truth in each group. The zero-shot student F1 is shown for comparison.
+ Datasets: CleanCoNLL, BC5CDR, MIT Movie, Hallmarks, MASSIVE en-US. CrossRE is not included, because its student task is being redesigned.

## 3. Results

**Teacher micro-F1 against ground truth, by student confidence decile** (1 = least sure):

| Dataset | Pool | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | Random 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CleanCoNLL | 13,957 | .503 | .788 | .784 | .785 | .777 | .777 | .824 | .800 | .832 | .783 | .799 |
| BC5CDR | 5,228 | .282 | .715 | .725 | .766 | .790 | .770 | .809 | .844 | .877 | .886 | .784 |
| MIT Movie | 8,797 | .714 | .689 | .684 | .711 | .742 | .741 | .775 | .785 | .805 | .801 | .726 |
| Hallmarks | 12,119 | .476 | .448 | .454 | .428 | .435 | .434 | .445 | .386 | .407 | .399 | .434 |
| MASSIVE en-US | 11,514 | .497 | .492 | .508 | .494 | .528 | .530 | .517 | .514 | .473 | .363 | .498 |

**Zero-shot student micro-F1, by decile:**

| Dataset | D1 | D5 | D10 | Random 10% |
|---|---:|---:|---:|---:|
| CleanCoNLL | .000 | .493 | .753 | .562 |
| BC5CDR | .000 | .605 | .865 | .633 |
| MIT Movie | .322 | .509 | .744 | .528 |
| Hallmarks | .167 | .116 | .166 | .125 |
| MASSIVE en-US | .278 | .305 | .291 | .291 |

**Share of sentences with no ground-truth item in the least-sure 10%:**

| Dataset | Least sure 10% | Random 10% | Most sure 10% |
|---|---:|---:|---:|
| CleanCoNLL | .745 | .222 | .014 |
| BC5CDR | .904 | .289 | .046 |
| MIT Movie | .028 | .007 | .001 |
| Hallmarks | .740 | .757 | .801 |
| MASSIVE en-US | .212 | .321 | .722 |

## 4. Findings

1. **NER: the teacher is worst where the student is least sure.** On BC5CDR, teacher F1 rises from .282 (D1) to .886 (D10). MIT Movie rises from about .69 to .80 (D2 to D10). CleanCoNLL drops only in D1 (.503).
2. **The least-sure NER group is mostly empty sentences.** A sentence with no student prediction gets confidence 0, so min selection takes it first. In D1, 74.5% (CleanCoNLL) and 90.4% (BC5CDR) of sentences have no gold entity. There the teacher produces false positives, and teacher F1 falls. The `ner_no_prediction_sensitivity` block (empty predictions ranked last) tests exactly this case.
3. **Classification and slots: zero-shot confidence carries little signal.** The zero-shot student scores about .12 (Hallmarks) and about .29 (MASSIVE) in every decile. The teacher is not worse in D1. On MASSIVE, the most-sure decile is mostly empty (72.2%), and the teacher is worst there (.363).

## 5. What it means for the paper

+ For NER, teacher error and student uncertainty go together. This supports the paper's mechanism: min selection sends the teacher the sentences it labels worst.
+ The effect has two parts: (a) empty sentences, where the teacher adds entities that do not exist, and (b) a smooth trend in D2 to D10 on BC5CDR and MIT Movie.
+ The main runs must report the `no_prediction` sensitivity result next to the primary result for NER.
+ This analysis uses zero-shot confidence. It says nothing about final student scores.

## 6. Limits

+ Micro-F1 on groups with few gold items (BC5CDR D1 has 0.11 gold entities per sentence) depends on a small number of spans. We give no confidence intervals.
+ One teacher, one prompt version per dataset, and one run of the student.
+ The script is exploratory. It is not part of the frozen analysis pipeline.

## Changelog

- 2026-09-26 07:10 CEST - Correction note: these numbers use the old parser, which also placed teacher mentions inside words (design section 17, "Teacher span placement"). With the fix, MIT Movie pool teacher F1 is .776, not .742; the other datasets move by under .004. The paper numbers come from the analysis scripts on the fixed parse, not from this note.
- 2026-09-25 23:28 CEST - Created from the pool analysis.
