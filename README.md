# Active GLiNER

Train a small, fast extraction model on labels from an LLM, and let the small model choose which texts the LLM labels.

```
your texts ──> GLiNER2.5 scores each text ──> pick the texts it is least sure about
                                                        │
          small fine-tuned model <── LoRA training <── LLM labels only those texts
          (runs on a CPU, ~1 ms per sentence on a GPU)
```

Active GLiNER works for four tasks with one command:

| Task | Example output |
|---|---|
| `ner` | `{"drug": ["aspirin"], "dose": ["100 mg"]}` |
| `relations` | `(Marie Curie, Paris, physical)` between given mentions |
| `classification` | several labels per text, or none |
| `slots` | JSON slots from a command, for example `{"time": ["7 am"]}` |

The teacher is any OpenAI-compatible endpoint: a local llama.cpp or vLLM server, or a hosted API.

## Install

```bash
uv pip install active-gliner        # from PyPI (after release)
# or, from this repository
uv sync
```

Python 3.11 or newer. A GPU is recommended for training; prediction runs on a CPU.

## Quickstart: your own data

1. Put your texts in a file, one per line, or as JSONL with a `text` field.
2. Start any OpenAI-compatible LLM server, or use a hosted one.
3. Run:

```bash
active-gliner fit \
  --texts texts.jsonl \
  --task ner \
  --labels "drug:a medicine or chemical,dose:an amount of a drug" \
  --teacher-url http://localhost:8000/v1 --teacher-model my-llm \
  --n 200 --select min \
  --out my-model
```

+ `--labels` - label names, each with an optional short definition after `:`.
+ `--n` - how many texts the LLM labels.
+ `--select` - `min` picks the texts the student is least sure about. `random` picks at random.
+ `--dev` - an optional file with correct labels. Without it, the report shows "teacher agreement", which is not accuracy.
+ `--dry-run` - prints the plan without loading a model.
+ `--api-key-env NAME` - for hosted teachers, reads the key from that environment variable (or from `.env`).

The output folder has the LoRA adapter, a one-page `report.md`, training curves, and every label the LLM gave.

Predict with the trained model:

```bash
active-gliner predict --adapter my-model/adapter/best --task ner --labels "drug,dose" --texts new.jsonl
```

## Follow a run

Every run writes plain files you can open anywhere:

| File | What it shows |
|---|---|
| `train_log.jsonl` | loss, learning rate, GPU memory and CPU at every step |
| `eval_log.jsonl` | dev loss and dev F1 at every check |
| `plots/training_curves.png` | loss (smoothed), dev F1 with the best point, learning rate, GPU and CPU |
| `plots/calibration.png` | how often predictions are right in each confidence band |
| `errors.jsonl` | every wrong prediction: missed, extra, wrong label or wrong boundary, with its confidence |
| `report.md` | a one-page summary |

`active-gliner watch <run folder>` prints the dev scores as a table while training runs.

## Reproduce the paper

The paper runs 309 experiments on six datasets. Every run goes through the same code.

```bash
active-gliner data check cleanconll          # load a dataset, freeze its splits, print a report
active-gliner label --dataset cleanconll --teacher gemma-4-12b --split pool
active-gliner tune --dataset cleanconll --trials 20 --freeze
active-gliner matrix list                    # the 309 runs, by block
active-gliner matrix run --block ner_core    # resumable: finished runs are skipped
active-gliner analyse                        # statistics, figures, paper/numbers.tex
```

Details:

+ Datasets: CleanCoNLL, BC5CDR, MIT Movie (corrected), CrossRE, Hallmarks of Cancer, MASSIVE 1.1 (English and French). Frozen split ids are in `data/splits/`. Raw data sources and versions are in `docs/research/2026-09-25-1230-dataset-sources.md`.
+ Student: `fastino/gliner2.5-multi-v1`, pinned by revision.
+ Teachers: Gemma 4 12B and Gemma 4 E4B (local, llama.cpp), Qwen 3.8 27B and gpt-oss-120b (Cerebras API). Launch scripts are in `scripts/`, pins in `configs/teachers/`.
+ The design, with every decision and review round, is in `docs/research/2026-09-25-1113-full-merge-design.md`.

## Development

```bash
just check          # ruff + tests
just run-smoke      # a real 20-step training run on the GPU
```

Rules for contributors and coding agents are in `AGENTS.md`.

## The thesis

This project grew out of the author's master thesis on MIT Movie. The thesis code and results stay available at the git tag `thesis-v1`. The full report is [here](https://drive.google.com/file/d/1eo1z6MbX-gSsD8jMPwdCOveldRVPxqrF/view?usp=drive_link). What the thesis did, and what the new study fixes, is listed in `docs/wiki/thesis-inventory.md`.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

Datasets, model weights (GLiNER, GLiNER2, and others), and LLM outputs are not covered by this licence. Each keeps the licence and terms of its owner. Check them before you reuse or redistribute data or labels.
