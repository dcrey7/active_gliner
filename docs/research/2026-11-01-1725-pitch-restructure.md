---
title: The paper as a pitch - structure around the owner's questions
date: 2026-11-01 17:25 CET
author: Abhishek Thomas
type: research
status: frozen
---

# The paper as a pitch

## 1. Goal

Endorsers and reviewers skim. The paper must give the idea in one page and hold
nothing wrong. No new experiments and no new charts; every number stays a macro.

## 2. Decisions (owner, 1 Nov 2026)

1. Title: "Active GLiNER: How to Mix Human and LLM Labels to Train a Small Extraction
   Model". The small model is the subject.
2. Synthetic data: one sentence citing the thesis (Fig 5.2.1: synthetic text and labels
   could not replace human examples). No figure, because the thesis protocol leaked
   test data into dev.
3. The headline answer is the per-task mix (Table 6, safe LLM share at a two-point
   loss), not "a quarter human". "About a quarter beats the LLM" stays as the second
   answer.

## 3. Structure

+ Abstract and a "findings at a glance" table: one row per question.
+ Introduction: why LLM labels, why not synthetic text, the five questions.
+ Figure 1: numbered steps 1 to 6 of one round; a dashed arrow marks the second round
  as not studied.
+ Results, one section per question:
  1. Which examples should the LLM label? (selection; teacher error by confidence)
  2. Which LLM? A bigger one, another family? (E4B vs 12B; Gemma vs Luna)
  3. How many labels, and how many human? (mixing curves, minimum human labels)
  4. What mix does each task need? (tolerance table, teacher errors)
  5. Is it worth it? (speed and cost)
+ Conclusion: the recipe as a per-task table.

## 4. Tool

The default human share follows the task type, from the same table: decision 0.5,
NER 0.75, multi-label 0.75, relations 1.0, slots 1.0. `--human-share` still overrides.

## Changelog

- 2026-11-01 17:25 CET - Created.
