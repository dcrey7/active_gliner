---
title: Paper figures redrawn and three review passes
date: 2026-10-29 01:39 CET
author: Abhishek Thomas
type: update
status: frozen
---

# Paper figures redrawn and three review passes

## 1. What was done

+ One figure style for the whole paper (`src/active_gliner/analysis/style.py`),
  modelled on Figure 2 of EmbeddingGemma 2 (arXiv 2610.37364): serif text that matches
  the Times body, a soft slate frame, a light dashed grid, thick lines with
  white-edged markers, shaded seed bands, and names written next to points.
+ Every figure is drawn at the size it prints (ACL column 3.03 in, page 6.3 in) and
  saved at exactly that size, so text in a figure prints at its set size (8 to 9.5 pt).
+ One legend above each multi-panel figure; shared axis titles; a short last row is
  centred, so no panel slot is left empty. Panel titles name the task type.
+ Mixing curves show three shares (LLM only, 25% human, humans only) and the
  teacher. The 50% and 75% lines and the random lines were dropped for clarity; the
  tables keep every share.
+ Minimum-human-labels figure marks the dev-chosen cell with a star. Quadrant chart:
  plain markers, every point named, no legend box. Teacher bins: coloured by task,
  named at the line end. Forest: coloured by task, value printed.
+ Figure 1 (overview) redrawn: filled rounded boxes in the figure palette, numbered
  marks where each question acts.
+ Colours checked with the dataviz palette validator: the ordinal blue ramp and the
  four task hues pass the colour-blind checks.

## 2. Review passes

+ Content: abstract to appendix against figures and tables. Fixed an unclear
  sentence on dev-chosen cells, the audit caption (now names GPT-6 Luna), and the
  audit intro (points to the added-sets paragraph).
+ Consistency: results subsections are all questions; results paragraph headings
  all state the finding; numbers use 1{,}000 everywhere; the tolerance table prints a
  true minus sign like the other tables.
+ Figures: captions match the new marks (bands, star, three shares); decision curves
  now sit side by side like the UNER curves, which removed a near-empty float page.

## 3. What was measured

+ `numbers.tex`, `results.json` and every table except `tolerance.tex` unchanged after
  regeneration; `tolerance.tex` changed only by math-mode minus signs.
+ Paper: 18 pages, no `??`.
+ `just check`: 343 passed, 1 skipped (GPU tests included).

## 4. What stays open

+ The abstract says adapters are released. 2,402 adapters (58 GB) sit under `runs/`;
  none is published. The owner decides: upload a curated set to Hugging Face, or
  reword the abstract.

## 5. What changed on disk

+ `src/active_gliner/analysis/{style,figures,curves,minimum,reliability,tolerance}.py`
+ `paper/main.tex`, `paper/figures/*`, `paper/tables/tolerance.tex`
+ `tests/test_style.py`, `tests/test_tolerance.py`

## Changelog

- 2026-10-29 01:39 CET - Created.
