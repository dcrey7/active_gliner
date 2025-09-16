Active GLiNER – Modular Active Learning Pipeline for NER
========================================================

Business Objective
------------------
- Build a simple, experiment‑friendly application to simulate an active‑learning loop for NER using GLiNER.
- Workflow (business simulation):
  - Run zero‑shot predictions on incoming business data (here: MIT Movie dataset).
  - Identify and surface the lowest‑confidence examples to humans for correction.
  - Use corrected labels to analyze domain patterns, optionally generate targeted synthetic data, and fine‑tune the model with LoRA.
  - Evaluate true model performance on a held‑out test set (no leakage) and estimate minimal corrected/synthetic data required for useful lift.

What’s Inside
-------------
- Modular code under `src/` (no heavy logic in `main.py`).
- Notebook‑friendly facade: `src/experiment/active_gliner.py` exposes small, easy wrappers:
  - Config: `load_config`, `ensure_dirs`
  - Data: `load_dataset`
  - Baseline (train subset): `compute_train_subset_results` (uses enhanced evaluation only)
  - Selection: `select_low_confidence`
  - Analysis + cache (Ollama): `get_or_create_analysis`
  - Synthetic + cache (Ollama): `get_or_create_synthetic`
  - Model/Train/Eval: `build_model`, `train_with_synthetic`, `evaluate_on_test`
  - Optional: `run_single_experiment` (end‑to‑end single run)
- Config‑driven hyperparameters via `config/default.json` (paths, LoRA, training, analysis, synthetic, evaluation).
- Enhanced evaluation only: `src/evaluation/enchanced_eval.py` (spelling kept to match existing file) provides overall F1 and overall confidence.

Key Design Choices
------------------
- Only enhanced evaluation is used for any metrics (train subset for selection; test set for final evaluation). The older quick_eval is not used.
- No data leakage: selection uses a fraction of the train split; final performance is evaluated on the test split.
- LoRA and Trainer hyperparameters are fully configurable (and overrideable from a notebook) with sensible defaults.
- Ollama analysis/synthetic generation are modularized and invoked via cached helpers.

Quick Start (Notebook)
----------------------
1) Import the facade
   - `import active_gliner as ag`
2) Load config and ensure directories
   - `cfg = ag.load_config()`
   - `ag.ensure_dirs(cfg)`
3) Load dataset
   - `train, test, labels = ag.load_dataset(cfg)`
4) Compute train‑subset baseline results (for selection only)
   - `results = ag.compute_train_subset_results(cfg, train, labels)`
5) Select lowest‑confidence examples to “send for correction”
   - `low = ag.select_low_confidence(results, n=cfg['experiment']['num_corrected'])`
6) (Optional) Domain analysis with LLM and synthetic data generation
   - `analysis = ag.get_or_create_analysis(cfg, low, labels)`
   - `final_summary = analysis.get('final_summary') if analysis else None`
   - `synthetic = ag.get_or_create_synthetic(cfg, low, labels, final_summary)`
7) Prepare training data and train with LoRA
   - Mix corrected examples with synthetic (as you prefer; see `run_single_experiment` for a reference pattern)
   - `model, monitor = ag.train_with_synthetic(cfg, train_data, val_data, device)`
8) Final evaluation on test set (enhanced)
   - `test_results = ag.evaluate_on_test(cfg, model, test, labels, device)`
   - Read: `test_results['overall_metrics']['overall_f1']` and `['overall_confidence']`

Configuration
-------------
- See `config/default.json` for:
  - `io`: data/results/logs/models paths
  - `model`: base model, `max_len`, `seed`
  - `lora`: `r`, `lora_alpha`, `lora_dropout`, `bias`, `task_type`, `target_modules`
  - `training`: steps, batch sizes, LR, scheduler, patience, precision, etc.
  - `evaluation`: `threshold`, `batch_size`, `train_subset_fraction`
  - `analysis`: batch size, retries, LLM model + params
  - `synthetic`: `num_synthetic`
  - `experiment`: `num_corrected`, `skip_analysis`

Repo Structure (Highlights)
---------------------------
- `src/experiment/active_gliner.py` — Simple API for notebooks (single‑run orchestration + wrappers).
- `src/preprocess/` — data loading + tokenization/transforms.
- `src/experiment/active_learning.py` — selection utilities.
- `src/ollama_calling/domain_analysis.py` — batch analysis + final summary (Ollama).
- `src/ollama_calling/synthetic_generation.py` — targeted prompts + incremental generation (Ollama).
- `src/ollama_calling/cache_combination.py` — caching glue for analysis/synthetic.
- `src/lora/` — LoRA config + application.
- `src/training/` — trainer, monitor, memory cleanup.
- `src/evaluation/enchanced_eval.py` — enhanced evaluation for F1 + confidence and rich analysis.

Recent Updates
--------------
- 2025‑09‑08 14:35 — Modularization + config pass
  - Added `src/experiment/active_gliner.py` facade with wrappers for config, data, selection, analysis, synthetic, training, and evaluation.
  - Moved Ollama functions out of `main.py` into:
    - `src/ollama_calling/domain_analysis.py` (batch analysis, combine, final summary)
    - `src/ollama_calling/synthetic_generation.py` (prompt builder, incremental generation)
  - Introduced `config/default.json` with configurable I/O, model, LoRA, training, evaluation, analysis, synthetic, and experiment settings.
  - Updated `src/training/trainer.py` to read LoRA and TrainingArguments from config, with safe defaults.
  - Extended `src/lora/lora_parameters.py` with `get_lora_config_from_dict` and optional config override in `apply_lora_to_model`.
  - Ensured evaluation uses only `enchanced_eval.enhanced_evaluate` for F1 + overall confidence (no quick_eval usage).
  - Left tests intact; no breaking changes to existing modules.

Notes
-----
- `test/src` contains package stubs used earlier for imports; with the new facade/package layout it can be removed eventually.
- The enhanced evaluation file name keeps the original spelling (`enchanced_eval.py`) for compatibility with existing imports.
- `main.py` remains as-is; prefer using the new `experiment.active_gliner` API in notebooks for clarity.
