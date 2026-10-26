---
title: Decision tasks (Jev, Unsloth, typed-decisions) and Active GLiNER
date: 2026-10-18 13:37 CEST
author: Claude
type: research
status: done
---

# Decision tasks (Jev, Unsloth, typed-decisions) and Active GLiNER

## Question

Is a "decision model" task a new task type for the paper, or classification?

## Findings

+ Decision model - a model that reads an input ("state") and answers typed questions
  with a probability over options, instead of writing text (Unsloth guide).
+ Three question types - `noul` (yes/no), `choice` (one of N), `score` (ordered levels)
  (TypeSafe System One schema, used by typed-decisions and Unsloth).
+ Unsloth training data - ag_news, arc, banking77, boolq, clinc150, commonsense_qa,
  mmlu, mnli, prompt_injections, snli, sst5, wanli, typed-decisions. Its headline test
  set: 2,000 typed-decisions + 500 BANKING77 + 500 CLINC150.
+ typed-decisions (LocalLLaMA) - 4 workflows (agent traces, customer service, invoices,
  security alerts), 300 train / 100 test cases each, 5 questions per case. Gold is the
  mean of three samples from a ~4B LLM teacher, not human labels. Scored on accuracy
  and calibration (KL, Brier).
+ Jev Decision Index - 37 public benchmarks; the sentence-level decision ones with human
  labels include VAST, FinEntity, iSarcasmEval, ANLI, BANKING77, CLINC150.

## Answer

+ Mechanics - classification. GLiNER2.5 answers `noul` and `choice` with its
  `classify_text` head; `score` is classification over the levels.
+ Task family - different from our Hallmarks task in four ways: single answer (not
  multi-label), ordered levels, often long structured input, and calibration is the
  metric. Calibration matters to our method, which ranks by confidence.
+ typed-decisions does not fit the study - its gold is LLM labels, so "human vs LLM
  label share" cannot be measured. Its 400-token+ JSON states may also exceed the
  student's context.
+ Fitting datasets (human labels) - SST-5 (score), BoolQ (noul), CLINC150 or VAST
  (choice). Avoid BANKING77 and SST-2: GLiNER2 reports zero-shot results on them.

## Every benchmark checked (Jev Decision Index, jevbench, open-system-one, Unsloth)

Fit tests: human labels, a labelling task (not an exam), short input, pool of about
5,000 or more. Sizes are approximate, from the dataset cards.

| Verdict | Benchmarks | Why |
|---|---|---|
| Fits, yes/no | BoolQ (~12.7k), iSarcasmEval (~4.9k) | human labels, short input |
| Fits, choice | CLINC150 (~23.7k), VAST stance (~18k), Amazon ESCI (large), ANLI (~163k, NC licence) | human labels, short input |
| Fits, score | SST-5 (~11.8k) | human ordered sentiment |
| Leakage risk | BANKING77, SST-2 | GLiNER2 reports zero-shot results on them |
| Not human labels | typed-decisions (LLM teacher gold), Emotion (hashtag labels), AG News (news-section metadata), RouterBench (model outputs), Phishing gradient (generated) | answer key is not a human label |
| Exam, not labelling | MMLU, GPQA, ARC, WinoGrande, HellaSwag, GSM8K, ChessBench, MuSR, SATA-Bench, CRUXEval, CLadder, HLE, OpenBookQA, CommonsenseQA | a student cannot learn knowledge from 400 labels |
| Too long | ContractNLI (whole NDAs), NLI4CT (trial reports), SGD (dialogues), email spam | beyond the encoder context |
| Too small | FinEntity (979), Support-ticket (300), prompt_injections (~600) | no room for pool, dev and test |
| Not text labelling | BFCL, ToolRet, API-Bank, Home simulator, BRIGHT, BPoMP, Humicroedit, POP909, cfcolor, ForecastBench, Habermas, 8 games | tools, retrieval, music, colour, forecasts |
| Extraction, not decision | ACOS (aspect quadruples) | fits the existing NER/relations tasks |

Special case: ANLI is written to fool models, so the LLM teacher is weak by design. It
tests the paper's question "when does teacher error erase the benefit" directly.

## Recommendation

v2 of the paper: one decision dataset per type (SST-5, BoolQ, CLINC150), same protocol.
Not for this submission: one dataset is about 200 runs, 2 to 3 days on one GPU lane.

## Sources

+ https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
+ https://huggingface.co/datasets/LocalLLaMA/typed-decisions
+ https://huggingface.co/spaces/multimodalart/jev-decision-index

## Changelog

- 2026-10-18 13:37 CEST - Created.
