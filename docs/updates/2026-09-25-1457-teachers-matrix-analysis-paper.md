---
title: Teachers, run matrix, analysis, paper draft
date: 2026-09-25 14:57 CEST
author: Claude (for Abhishek Thomas)
type: update
status: frozen
---

# Teachers, run matrix, analysis, paper draft

## 1. What I did

+ Step 5 teachers: OpenAI-compatible client, resumable cache, stats, prompts v1 and v2, validators; Gemma 4 12B labelling server (llama.cpp, 8 slots).
+ CrossRE redesigned as relation prediction over given mentions (design section 15, Codex round 25).
+ Prompt gate: v2 (label definitions) kept for every dataset except Hallmarks.
+ Step 6: 309-run matrix, resumable runner, routing and mixing, diversity selector, no-prediction sensitivity, Optuna tuning on dev with working pruning.
+ Step 7: analysis package (paired bootstrap, Holm, ASO, primary interaction, teacher bins, costs, Pareto and quadrant figures, LaTeX number macros).
+ Found that Qwen 3.8 2B does not exist; Gemma 4 E4B replaces it (design section 16, Codex round 26). E4B GGUF downloaded and pinned.
+ Bring-your-own-data commands `fit` and `predict`; new README.
+ Paper draft in `paper/` (ACL preprint template, Tectonic build), bibliography checked against arXiv and the ACL Anthology.
+ O4 overlap audit by a subagent.
+ Safety: `.env` added to `.gitignore`; `.env` loaded at CLI start; missing keys give a clear error.

## 2. What I measured

+ Gemma 4 12B speed alone: 5.0 sentences per second (8 slots), 64/64 valid JSON; about 2.2 per second while tuning shares the GPU.
+ Prompt gate (200 dev, micro-F1, v1 -> v2): CleanCoNLL 0.780 -> 0.794, BC5CDR 0.801 -> 0.842, MIT Movie 0.707 -> 0.727, CrossRE 0.146 -> 0.269, Hallmarks 0.453 -> 0.439 (v1 kept), MASSIVE en 0.501 -> 0.573, fr 0.433 -> 0.468.
+ Tuning CleanCoNLL trial 0 (vendor-inspired recipe, N 400 ground truth): best dev micro-F1 0.859. About 10 minutes per trial with the GPU shared.
+ Tests: 101 unit tests pass without GPU; the GPU suite passed at 127 before the later additions.

## 3. What failed or stays unknown

+ The Gemma server crashed after 94 minutes: `CUDA error: an illegal memory access was encountered`, kernel `Xid 13` and `Xid 43` for llama-server at 14:46. The clock cap (1800 MHz) was active. Likely cause: the MTP speculative drafter with 8 parallel slots. Fix: no drafter, keep-alive launcher, client retries and waits for the server. Labelling restarted at 14:55 and resumes from the cache.
+ No Cerebras API key on the machine: Qwen 3.8 27B and gpt-oss-120b parts are blocked (about 39 runs plus ladder labelling).
+ No official Qwen 3.8 2B exists (fixed by section 16).
+ Not yet measured: full timing gate (O7), same-seed repeatability.

## 4. What changed on disk

Branch `worktree-v2-research-plan` (local), commits `4378064` to `e5b119b`:

+ `src/active_gliner/teachers/`, `matrix.py`, `mixing.py`, `tune.py`, `byod.py`, `analysis/`
+ `configs/teachers/*.yaml`, `configs/prompts.yaml`, `configs/label_definitions.yaml`, `configs/prices.yaml`
+ `scripts/serve_teacher_gemma.sh`, `scripts/serve_teacher_gemma_e4b.sh`, `scripts/keep_alive.sh`
+ `paper/` (main.tex, references.bib, references_extra.bib, acl.sty, acl_natbib.bst)
+ docs: design sections 15-16, prompt gate note, overlap audit, this update
+ Outside the repo: `~/models/gemma-4-e4b-qat/`, `~/.local/bin/tectonic`, labels in `labels/` (git-ignored)

Running at 14:57: Gemma labelling of all pool and test splits; tuning (4 tasks x 20 trials).

## 5. What the next session must do first

1. Check `/home/abhishek/.claude/jobs/b454a3ee/tmp/label_all2.log` and `runs/tuning/*/trials.jsonl`; restart either job if it died (both resume).
2. When tuning ends: check `configs/recipes/*.yaml`, run the O7 timing gate and the same-seed repeatability check.
3. Start `active-gliner matrix run` with ground-truth blocks first (they need no teacher labels).
4. Ask Abhishek for the Cerebras key if still missing.

## Changelog

- 2026-09-25 14:57 CEST - Created.
