# AGENTS.md

Rules for every coding agent in this repo. Read this file first.

## 1. The project

Active GLiNER studies one question: when does student uncertainty improve one-round LLM annotation, and when does teacher error erase that benefit?

+ Student: GLiNER2.5-multi (`fastino/gliner2.5-multi-v1`, `gliner2` package) with LoRA.
+ Teachers: local and API LLMs that label the selected sentences.
+ Tasks: NER, relation extraction, multi-label classification, slot JSON.
+ Output: an arXiv paper and a `pip install active-gliner` tool.

The design lives in `docs/research/2026-09-25-1113-full-merge-design.md`. The objectives live in `docs/wiki/objectives.md`. Do not change the design in code. If the code needs a design change, stop and say so.

## 2. Who does what

+ Codex writes the code in `src/`, `justfile`, `pyproject.toml`, and configs.
+ Claude reviews the code, writes tests in `tests/`, and writes `docs/`.
+ A change is done only when `just check` passes and Claude has reviewed it.

## 3. Layout

| Path | Content | Status |
|---|---|---|
| `src/active_gliner/` | the new package on `gliner2` | active |
| `tests/` | pytest tests for `src/` | active |
| `configs/` | one config file per paper run type | active |
| `docs/` | wiki, research, updates | active |
| `src2/`, `test/`, `main.py`, `requirements.txt` | thesis code (git tag `thesis-v1`) | read-only; port parts, never edit |
| `results2/`, `data/` | thesis results and data | read-only |

The thesis code uses old `gliner` and `torch<2.5`. It does not run in the new environment. To run it, check out the `thesis-v1` tag.

## 4. Toolchain

| Job | Tool | Command |
|---|---|---|
| Packages | `uv` | `uv add`, `uv sync`. Never `pip`, never `conda`. |
| Run | `uv run` | never activate a venv by hand |
| Lint and format | `ruff` | `just lint`, `just fmt` |
| Tests | `pytest` | `just test` |
| All checks | `just` | `just check` (lint + format check + tests) |

Put every long command in the `justfile`. Never run `sudo`.

## 5. Code style

+ Write simple code. A junior engineer must understand it on one read.
+ Small functions. Plain names. Type hints on public functions.
+ No clever abstractions. No base class with one child.
+ Comments explain why, not what.
+ No new dependency without a reason in the commit message.

## 6. Research rules (do not break these)

1. Three splits per dataset: pool (train), dev, test. Split ids are written to disk once and then frozen.
2. Dev is for every choice: tuning, early stopping, thresholds, prompts.
3. Test is for final scores only. No code path may tune on test.
4. Never filter the pool with ground-truth labels. Sentences with no entity, relation, label, or slot stay in the pool.
5. All arms of one comparison use the same frozen pool ids and matched seeds.
6. A seed sets pool tie-breaks, data order, and LoRA initialisation.
7. Every run logs: config, seed, pins (model revisions, library versions, dataset revisions), scores, wall time, GPU time.
8. Teacher labels are cached on disk, resumable, and validated (JSON, schema, exact substring, allowed types).
9. Never give ground truth, gold intents, or hints to a teacher.
10. Numbers in the paper come from logs through a script. Never type a result by hand.

## 7. Safety

+ Never commit a token, a key, or a password. Secrets live in `.env` (see `.env.example`).
+ Never upload to Hugging Face or any other service without the owner's word.
+ Never push to `main` or `master`. Never force-push.
+ No "Co-Authored-By" or "Generated with" lines in commits or pull requests.

## Changelog

- 2026-09-25 11:55 CEST - Created for O1.
