---
title: Refactor on gliner2, build steps 1-4
date: 2026-09-25 13:05 CEST
author: Claude (for Abhishek Thomas)
type: update
status: frozen
---

# Refactor on gliner2, build steps 1-4

## 1. What I did

+ O1 repo setup: `AGENTS.md` (Codex writes code, Claude reviews), `uv`, `ruff`, `pytest`, `justfile`. Thesis code tagged `thesis-v1` (local tag).
+ Studied `gliner2` 2.0.0 hands-on on the 3090, scanned the Fastino platform, and listed the thesis observability code.
+ Student changed to `fastino/gliner2.5-multi-v1` (Abhishek); MASSIVE fr-FR added; design section 13 (Codex rounds 20-21).
+ Refactor plan signed off by Codex (rounds 22-24).
+ Step 1: pinned loader from a snapshot, `pins.json`.
+ Step 2 (O2 done): 7 loaders, frozen split ids, data reports; design section 14.
+ Step 3: metrics, greedy flat decoder, confidence rules, selection; 4 task plug-ins.
+ Step 4: one command runs select, LoRA train, best checkpoint, dev and test eval, and writes live logs, plots, calibration, errors, report card.

## 2. What I measured

+ Tests: 93 pass (`just check`, GPU, 40 s).
+ Split sizes (pool / dev / test): CleanCoNLL 13,957 / 3,233 / 3,427; BC5CDR 5,228 / 5,330 / 5,865; MIT Movie 8,797 / 978 / 2,443; CrossRE 2,519 / 300 / 2,446; Hallmarks 12,119 / 1,798 / 3,547; MASSIVE en-US and fr-FR 11,514 / 2,033 / 2,974.
+ Gold spans not on word boundaries (dev): 0.0% on every dataset, so no custom word splitter.
+ CrossRE: 25% of sentences have no relation; 6.5% of relation arguments repeat in their sentence (above the 2% rule, so a without-repeats view is reported).
+ Student speed (hands-on study): about 1 ms per sentence at batch 32; 1.1-1.6 GB GPU.
+ Zero-shot micro-F1 on 32 dev records (CPU run by Codex): CleanCoNLL 0.52, BC5CDR 0.69, MIT Movie 0.51, CrossRE 0.05, Hallmarks 0.13, MASSIVE en 0.28, fr 0.21. Small samples; not paper numbers.
+ Smoke run (MIT Movie, 48 picked from 300, 20 LoRA steps): 15 s end to end; dev 0.617, test 0.584.

## 3. What failed or stays unknown

+ Same seed, two runs, different dev F1 at step 20: 0.625 vs 0.604. GPU training is not deterministic by default (`cudnn.benchmark`). O7 measures this and turns on deterministic mode if cheap.
+ The thesis `data/mit-movie/dev.json` is an exact copy of `test.json` (original MIT labels, not the Galileo fix).
+ Codex's sandbox has no GPU; GPU tests are run by Claude.
+ Not built yet: teachers (step 5), tuning, diversity selector, analysis scripts.

## 4. What changed on disk

Branch `worktree-v2-research-plan` (local, not pushed), commits `1f5b8d3` to `1da75bb`:

+ `AGENTS.md`, `pyproject.toml`, `uv.lock`, `justfile`
+ `src/active_gliner/`: `cli.py`, `model.py`, `pins.py`, `data/`, `tasks/`, `decode.py`, `confidence.py`, `selection.py`, `evaluate/`, `observe/`, `train.py`, `run.py`, `report.py`
+ `tests/`: smoke, model and pins, data, core logic, task plug-ins, run smoke
+ `data/splits/`, `data/reports/`
+ docs: gliner2 API study, Fastino platform scan, dataset sources, refactor plan, design sections 13-14, objectives
+ Raw data outside the repo: `~/.cache/active_gliner/raw/`

## 5. What the next session must do first

1. Step 5 (O6): teacher client (OpenAI-compatible, llama.cpp for Gemma 4 12B), label cache, stats, per-task prompts and validators; `labels_source="gemma"` in `run.py`.
2. Choose slot mode on en-US dev zero-shot (structure vs entity, 5-point rule).
3. Same-seed repeatability check, then the O7 timing gate.

## Changelog

- 2026-09-25 13:05 CEST - Created.
