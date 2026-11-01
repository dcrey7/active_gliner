---
title: Paper restructured around the five questions; tool default human share per task
date: 2026-11-01 17:53 CET
author: Abhishek Thomas
type: update
status: frozen
---

# Paper restructured around the five questions

Spec: `docs/research/2026-11-01-1725-pitch-restructure.md`.

## 1. What was done

+ New title - "Active GLiNER: How to Mix Human and LLM Labels to Train a Small Extraction
  Model".
+ Abstract - built around the questions; the headline is the per-task mix.
+ Introduction - why LLM labels, one sentence on LLM-written texts (thesis cited, new bib
  entry `thomas2025thesis`), the five questions, and Table 1 "Findings at a glance".
+ Figure 1 - six numbered steps (score, rank, top N, split, fine-tune, evaluate) and a dashed
  "second round: not studied" arrow.
+ Results - one subsection per question, each opening with a "Short answer":
  4.1 which examples, 4.2 which LLM (bigger E4B to 12B; other family Gemma to Luna),
  4.3 how many labels, 4.4 what mix per task, 4.5 is it worth it.
+ Conclusion - new title; the recipe now points to Table 9, the mix per task type.
+ Precision fixes found on the read-through:
  + Min selection beats random on relations even with Gemma labels (+1.37), so the abstract
    and answers now say "mostly not" and name the tasks.
  + Relations lose with LLM labels because the LLM is weak there, not because it marks
    extra labels; the abstract and the 4.4 answer say so.
+ No new experiments and no new figures.

## 2. New macros (rule 10)

+ `TaskType{Decision,NER,MultiLabel,Relations,Slots}SafeShare{Low,High,GemmaLow}` and
  `...DefaultHumanShare` (`analysis/tolerance.py`, `family_macros`).
+ `LadderBetterCount`, `LadderPlaces` (`analysis/tables.py`): the 12B teacher is better
  where the gap interval is above zero.
+ `\resrange` in `main.tex` prints one value when both ends are equal.

## 3. Tool

+ `byod.DEFAULT_HUMAN_SHARE` - decision 0.5, NER 0.75, classification 0.75, relations 1.0,
  slots 1.0. `rank` and `--human-share` use it when no share is given.
+ `fit` trains on human labels alone when `for_llm.jsonl` is empty; the LLM is not called.
+ A test checks the defaults against `paper/numbers.tex`.
+ README - new title, the five answers, the per-task table with the default, the two goals
  (close to all-human, or only beat the LLM with 0.25).
+ Notebooks - keep `human_share=0.25` (the "beat the LLM" goal) so the demos show both
  label sources; the text cell now says so. Saved outputs stay valid; no rerun.

## 4. What was measured

+ `numbers.tex` - 22 lines added, none changed. Figures and tables unchanged.
+ Paper - 19 pages (was 18), no `??`, every page read.
+ `just check` - 347 passed, 1 skipped (GPU tests included).
+ `dist/arxiv.tar.gz` - 2.7 MB, builds.

## 5. What stays open

+ The abstract no longer claims released adapters. The owner decides whether to upload a
  curated set to Hugging Face.
+ Old Co-Authored-By trailer in master commit 27167f7: history rewrite needs the owner's OK.

## Changelog

- 2026-11-01 17:53 CET - Created.
