---
title: How few human labels beat the LLM teacher - design draft for review
date: 2026-10-02 12:07 CEST
author: Claude (for Abhishek Thomas)
type: research
status: draft, for Codex review
---

# How few human labels beat the LLM teacher

## 1. The question

The thesis claim (thesis section 4.6, Fig 4.6.1, MIT Movie only):

+ Student on human labels only = upper bound.
+ Student on LLM labels only = lower bound; it reaches the LLM's own F1.
+ Mixing human and LLM labels gives a student better than the LLM, with fewer human labels.

The new question: for each dataset, what is the **minimum number of human labels** (and minimum human share) at which the student beats the LLM teacher? Where does the curve stop rising?

## 2. What exists (as of 2 Oct 2026)

+ Mixing at one budget only (N = 400; CrossRE 200), min selection, shares 0/25/50/75/100%, 3 seeds.
+ At N = 400 the student beats the teacher on MIT Movie, MASSIVE en/fr, CrossRE (from 50%), but not on BC5CDR, CleanCoNLL, Hallmarks: there even 100% human with min selection stays below the teacher (BC5CDR 72.4 vs teacher 79.0).
+ Cause: the new protocol keeps sentences with no entity in the pool (research rule 4). min ranks no-prediction sentences first, so at N = 400 the set is mostly empty sentences (BC5CDR has 810, CleanCoNLL 1,803, Hallmarks 10,545). The thesis pool filtered them out with ground truth, so its min picks always had entities.
+ Random selection with 400 human labels beats the teacher on BC5CDR (82.5), CleanCoNLL (85.7), Hallmarks (43.0 vs 42.9).
+ Ground-truth min curve exists at N = 50, 100, 200, 400, 1000; whole pool for random-free "all".
+ Pool sizes: CleanCoNLL 13,957; BC5CDR 5,228; MIT Movie 8,797; CrossRE 2,519; Hallmarks 12,119; MASSIVE en/fr 11,514.
+ Gemma labels: whole pool and test for every dataset; dev fully only for MIT Movie (others 4 to 67%).
+ Run time: 10 to 60 min per run (median about 15 to 30), nearly flat in N; 3 GPU lanes give about 8 runs/h.

## 3. Proposed design

### 3.1 Grid

+ Budgets N: 50, 100, 200, 400, 1000, 2500, 5000, whole pool (log-spaced; capped at the pool size). 2500 is not assumed to be the limit.
+ Human share r: 0, 25, 50, 75, 100%. Same N sentences for every r; only the label source changes (thesis design). Human sentences inside the N are a seeded random subset.
+ Selection rules:
  + A. random - primary for this question (no selection effect).
  + B. min with no-prediction sentences last - the thesis's intent without a ground-truth filter.
  + C. min (thesis literal) - only the cells that already exist; no new runs.
+ Seeds: 3.

### 3.2 Reading the minimum

+ Plot F1 against the number of **human labels** H = r x N (LLM labels are almost free), one line per r, with the teacher's F1 as a horizontal line. Also the thesis view: F1 against N, one line per r.
+ The teacher threshold and the minimum are found on **dev**: teacher dev F1 needs Gemma labels on the dev splits (about 1 to 2 GPU hours with the local server).
+ Minimum H*: the smallest grid point whose seed-mean dev F1 is above teacher dev F1 and whose paired-bootstrap lower bound is above it too. Then report test F1 at H* once.
+ Ceiling: whole-pool human (exists) and the point where the curve flattens.

### 3.3 Cost

Per dataset and rule: 8 budgets x 5 shares x 3 seeds = 120 cells, minus existing ones. Two rules x 7 datasets is about 1,500 runs, about 8 days. Too many; options to cut in 4.

## 4. Open questions for the review

1. Is fixed-N mixing the right design, or is a **fixed human core plus LLM labels for the rest** better ("I have H human labels; label the rest of the pool with the LLM")? That is one line per H with L = pool - H, much cheaper (about 7 H values x 3 seeds per dataset) and closer to how a team would work.
2. Which selection rule should be primary for the minimum: random, or min-empty-last?
3. Which cuts keep the answer valid: fewer datasets (drop MASSIVE fr?), fewer shares (0, 25, 50, 100), fewer budgets, 2 seeds at large N?
4. Is "smallest grid point whose bootstrap lower bound beats the teacher, chosen on dev" a sound minimum, or should we fit a saturating curve F1(H) = a - b H^-c and interpolate?
5. Teacher dev F1 vs teacher test F1: the teacher is not trained, so is comparing on test acceptable, or must the crossing be chosen on dev?
6. Anything in the protocol (frozen pool, seeds, early stopping on dev, test once) that this design breaks.

## Changelog

- 2026-10-02 12:07 CEST - Draft for Codex review.
