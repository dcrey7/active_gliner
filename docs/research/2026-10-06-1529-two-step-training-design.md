---
title: Two-step training - teacher labels first, then human labels
date: 2026-10-06 15:29 CEST
author: Claude (approved by Abhishek, 2026-10-06)
type: research
status: dropped
---

# Two-step training: teacher labels first, then human labels

## Why

The minimum-human-labels curve (design: `2026-10-02-1207-minimum-human-labels-design.md`)
shows a problem. With the same H human sentences, a student trained on the human
sentences alone beats a student trained on the human sentences plus teacher labels on
the rest of the pool, on most datasets and counts:

+ CleanCoNLL - human alone wins at every count, by 3 to 14 points.
+ CrossRE - human alone wins from H = 50, by 4 to 16 points.
+ BC5CDR - teacher labels help at H = 50 and 100, hurt from H = 200.
+ Hallmarks - teacher labels help up to H = 200, hurt from H = 400.

In one batch, thousands of teacher labels outvote a few hundred human labels, so the
student learns the teacher's mistakes. Two-step training is the common fix for noisy
labels: learn the broad task from the noisy labels, then correct it on clean labels.

## Design

+ Step 1 - the teacher-only whole-pool run that already exists
  (`all-seed{s}-long`, labels gemma-4-12b). No new training.
+ Step 2 - merge the step-1 adapter into the student weights, then train a fresh LoRA
  on the same nested H human sentences that the minimum curve uses for that seed.
+ Grid - every minimum-curve cell: 7 dataset-locale places, H in
  {25, 50, 100, 200, 400, 1000, 2500} below the pool size, seeds 1 to 3. 147 runs.
+ Step 2 uses the frozen recipe (step cap, early stopping on dev), like the
  human-only runs it is compared with.
+ Run folder - `all-seed{s}-gt{share}nested-two-step` under the teacher folder. The
  pins record the step-1 run and a hash of its adapter files.

## Comparisons

Same seed, same H human sentences:

1. Two-step against mixed in one batch (the minimum curve) - same human ids exactly.
2. Two-step against human only (random N = H) - same count, different random draw.
3. Two-step against the teacher - the dev-chosen minimum rule of the minimum design.

## What it can and cannot show

+ It can show that teacher labels add value on top of human labels when the order of
  training changes.
+ It cannot separate "two steps" from "step 2 sees fewer sentences": that is the
  method. The human-only line is the control for the second part.

## Changelog

- 2026-10-06 15:29 CEST - Created; runs queued after the open curve jobs.
- 2026-10-08 11:40 CEST - Dropped before any run: the paper tests ranked top-N selection, which the mixing curves cover.
