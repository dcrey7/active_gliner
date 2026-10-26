---
title: Design addendum - decision tasks (yes/no, choice, ordered levels)
date: 2026-10-18 13:45 CEST
author: Claude
type: research
status: accepted by the owner (18 Oct 2026, "this also has to be part of it")
---

# Design addendum - decision tasks

Adds a fifth task family to the study. Background and dataset choice:
`docs/research/2026-10-18-1337-decision-tasks.md`.

## 1. Task

+ Decision - one input, one answer from a fixed option list. Single-label classification.
+ Student - the same GLiNER2.5-multi with LoRA. Its `classify_text` head with
  `multi_label=False`. No Decide model.
+ Teachers - Gemma 4 12B (main) and GPT-6 Luna, same prompt for both. Validation: JSON
  schema, exactly one allowed option.
+ Metric - micro-F1 over decisions. With one gold and one predicted answer per input it
  equals accuracy. Paired bootstrap over test inputs, as for the other tasks.
+ Confidence - the top option probability. Least confident first. Every input gets an
  answer, so the empty-prediction rule does not apply.

## 2. Datasets

| Type | Dataset | HF repo | Input | Options | Pool / dev / test |
|---|---|---|---|---|---|
| yes/no | BoolQ | google/boolq | question + passage | yes, no | train minus 1,000 / 1,000 from train / validation (3,270) |
| choice | CLINC150 | clinc/clinc_oos, plus | utterance | 150 intents + out of scope | train / validation / test |
| levels | SST-5 | SetFit/sst5 | sentence | very negative ... very positive | train / dev / test |

+ BoolQ test - the HF test split has no labels, so the HF validation split is the test
  set and dev is a seeded 1,000-question hold-out from train (rule 1, frozen once).
+ Exposure - GLiNER2.5 reports zero-shot results on clinc_oos and rotten_tomatoes (the
  source text of SST). Disclosed in the audit appendix, like CrossRE.

## 3. Runs per dataset (one GPU lane)

+ Gemma curves - ranked min at N 100/400/1000/2500, human share 0/25/50/75/100; random
  0% and 100% at the same N; whole pool (long) at shares 0/25/50/75/100. 3 seeds.
+ Luna curves - the same ranked and whole-pool cells with Luna labels (shares 0 to 75),
  plus random Luna at N 400. 3 seeds.
+ Main-table cells - min and random at N 400 come from the curve cells.
+ About 160 runs per dataset, about 480 in total.

## 4. Analysis and paper

+ Decisions are secondary and descriptive. The pre-registered Holm families stay as
  they are. The dev-chosen fewest-human-labels test check runs with its own Holm family.
+ Recipe - the frozen classification recipe, not tuned again. Stated in the paper.

## Changelog

- 2026-10-18 13:45 CEST - Created.
