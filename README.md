# Active GLiNER

Train a small, fast extraction model on labels from an LLM, with a few human labels where they
matter most.

## Why

You want a model that finds names, relations, labels or answers in text, fast and cheap.

+ Human labels - accurate, but slow and costly.
+ An LLM on every input - no training needed, but slow, it costs money per call, and your data
  leaves your servers.
+ The middle path - let the LLM label the training data once, then train a small model on it.
  The small model runs on a CPU. In our tests it was 165 times faster than its LLM teacher at
  almost the same NER score (75.3 against 76.4 F1).

Active GLiNER does this with GLiNER2.5 as the small model. It also answers the question the
paper asks: **which examples should the LLM label, and how many should humans label instead?**

```
your unlabelled texts
   │  1. the untrained small model ranks them once, least sure first
   ▼
top N texts ──> a random quarter: human labels
           └──> the rest: LLM labels
   │  2. train the small model on both (LoRA)
   ▼
a small, fast model you own ──> 3. check it on labelled dev data ──> 4. predict
```

## Five task types

| Task | What it does | Example |
|---|---|---|
| `ner` | finds names and other short spans | `"drew barrymore"` = actor |
| `relations` | says how two given mentions relate | (Marie Curie, Paris) = physical |
| `classification` | gives a text all labels that apply, or none | a sentence gets `avoiding immune destruction` |
| `slots` | turns a command into JSON slots | `{"time": ["7 am"], "date": ["tomorrow"]}` |
| `decision` | picks exactly one answer from a list | `"how much is on my card"` = `balance` |

The LLM teacher is any OpenAI-compatible endpoint: a local llama.cpp or vLLM server, or a hosted
API.

## Install

```bash
uv pip install active-gliner        # from PyPI (after release)
# or, from this repository
uv sync
```

Python 3.11 or newer. A GPU is recommended for training (8 GB is enough); prediction runs on a
CPU.

## Notebooks

Each notebook runs the full recipe on a real dataset, with the outputs shown. It ranks 1,000
texts, gives 250 to humans (copied from the ground truth), lets Gemma 4 12B label the rest, and
trains for 1,000 steps. The last column is dev F1 from one run (seed 1):

| Notebook | Task | Data | Dev F1, zero-shot -> trained |
|---|---|---|---|
| [`task_ner.ipynb`](notebooks/task_ner.ipynb) | NER | MIT Movie questions, 12 entity types | 47.4 -> 74.6 |
| [`task_relations.ipynb`](notebooks/task_relations.ipynb) | relations | CrossRE, 17 relation types | 9.6 -> 21.4 |
| [`task_classification.ipynb`](notebooks/task_classification.ipynb) | multi-label classification | Hallmarks of Cancer, 10 labels | 21.0 -> 43.6 |
| [`task_slots.ipynb`](notebooks/task_slots.ipynb) | slots to JSON | MASSIVE commands, 55 slot types | 25.4 -> 63.6 |
| [`task_decision.ipynb`](notebooks/task_decision.ipynb) | decision | CLINC150 intents, 151 options | 68.7 -> 92.7 |
| [`quickstart.ipynb`](notebooks/quickstart.ipynb) | NER, LLM labels only | the shortest path: 200 LLM labels | |

## The recipe, step by step

**1. Rank once, and split.** The untrained model scores every text. `rank` takes the 1,000 it is
least sure about (texts where it finds nothing go last) and puts a random quarter of them aside
for humans.

```bash
active-gliner rank --texts texts.jsonl --task ner \
  --labels "drug:a medicine or chemical,dose:an amount of a drug" \
  --n 1000 --human-share 0.25 --out ranked
```

This writes `ranked/for_humans.jsonl` and `ranked/for_llm.jsonl`.

**2. Label the human share.** People label `for_humans.jsonl` by hand, in the same format as the
dev file (see below). Save it as `human.jsonl`.

**3. LLM labels for the rest, then train.**

```bash
active-gliner fit --texts ranked/for_llm.jsonl --select all --human human.jsonl \
  --task ner --labels "drug:a medicine or chemical,dose:an amount of a drug" \
  --dev dev.jsonl \
  --teacher-url http://localhost:8000/v1 --teacher-model my-llm \
  --out my-model
```

The LLM never sees the human-labelled texts. Every LLM answer is checked: valid JSON, allowed
labels, and spans that occur exactly in the text.

**4. Predict.**

```bash
active-gliner predict --adapter my-model/adapter/best --task ner --labels "drug,dose" --texts new.jsonl
```

No humans at all? Skip steps 1 and 2: `fit --texts texts.jsonl --n 400 --select random` lets the
LLM label 400 texts picked at random.

The same from Python:

```python
from active_gliner import byod

ranked = byod.rank("texts.jsonl", "ner", "drug,dose", n=1000, human_share=0.25, out_dir="ranked")
run = byod.fit(
    texts=ranked / "for_llm.jsonl", select="all", human="human.jsonl",
    task="ner", labels="drug,dose", dev="dev.jsonl",
    teacher_url="http://localhost:8000/v1", teacher_model="my-llm", out_dir="my-model",
)
predictions = byod.predict(run / "adapter/best", "ner", "drug,dose", "new.jsonl")
```

### Options

+ `--labels` - label names, each with an optional short definition after `:`.
+ `--n` - how many texts to rank or to label.
+ `--select` - `min` (least sure first, empty texts last), `random`, or `all` (every text in the
  file, for example `for_llm.jsonl`).
+ `--human` - a labelled file whose texts train the model with their human labels.
+ `--dev` - a labelled file to measure accuracy. Without it, the report shows "teacher
  agreement", which is not accuracy.
+ `--api-key-env NAME` - for hosted teachers, reads the key from that environment variable (or
  from `.env`).
+ `--dry-run` - prints the plan without loading a model.
+ `--max-steps`, `--eval-steps` - training length and how often dev is checked (1000 and 100, as
  in the paper; the best dev checkpoint is kept).
+ `--batch-size`, `--grad-accum` - change the batch when the GPU is short of memory. By default
  `fit` uses the paper's tuned settings for the task type (`src/active_gliner/recipes/`).

### File formats

One JSON object per line. Unlabelled texts need only `text`. Labelled files (dev, human) add:

```jsonl
{"text": "Take aspirin 100 mg daily .", "entities": [{"text": "aspirin", "label": "drug"}, {"text": "100 mg", "label": "dose"}]}
{"text": "Samples were stored at -80 C .", "labels": []}
{"text": "how much is on my visa card", "label": "balance"}
{"text": "Curie moved to Paris .", "entities": [{"text": "Curie", "label": "person"}, {"text": "Paris", "label": "location"}], "relations": [{"head": 0, "tail": 1, "type": "physical"}]}
```

+ `ner` and `slots` - `entities`, each with `text` and `label`, and optionally `start`/`end`.
+ `classification` - `labels`, a list (empty for none).
+ `decision` - `label`, exactly one.
+ `relations` - `entities` (the mentions, also needed in unlabelled texts) and `relations`, where
  `head` and `tail` are positions in the `entities` list.

## What the paper found

The paper is [`paper/main.pdf`](paper/main.pdf). It asks three questions on nine datasets with
two LLM teachers (Gemma 4 12B and GPT-6 Luna).

1. **Should the small model choose which texts the LLM labels?** Mostly no. Picking the texts it
   is least sure about does not beat random picking, mostly because it picks texts with nothing
   to find. That is why `rank` puts those texts last.
2. **How should human and LLM labels be mixed?** Rank once, give a random quarter of the top
   1,000 to humans and the rest to the LLM. A quarter or fewer human labels beats the LLM itself
   on 6 of 7 test sets.
3. **How many LLM labels can each task take?** It depends on the task:

| Task | LLM share that costs under about 2 F1 points | Why |
|---|---|---|
| decision | half, or more | one answer from a list with plain names |
| NER | a quarter | a few common entity types |
| multi-label classification | a quarter at most | the LLM labels many texts the expert marked as none |
| relations | less than a quarter | the LLM is weak on relations |
| slots | less than a quarter | the LLM fills slots in commands that have none |

Before you start, score the LLM on a few hundred labelled dev texts. If it often adds labels
where the dev set has none, give humans a larger share.

## Follow a run

Every run writes plain files you can open anywhere:

| File | What it shows |
|---|---|
| `report.md` | a one-page summary |
| `train_log.jsonl` | loss, learning rate, GPU memory and CPU at every step |
| `eval_log.jsonl` | dev loss and dev F1 at every check |
| `plots/training_curves.png` | loss, dev F1 with the best point, learning rate, GPU and CPU |
| `plots/calibration.png` | how often predictions are right in each confidence band |
| `errors.jsonl` | every wrong prediction: missed, extra, wrong label or wrong boundary |
| `teacher_stats.json` | how many LLM answers passed the checks |

`active-gliner watch <run folder>` prints the dev scores as a table while training runs.

## Reproduce the paper

Every number in the paper comes from the run logs through `active-gliner analyse` into
`paper/numbers.tex`. The study is a fixed matrix of runs; every run goes through the same code.

```bash
active-gliner data check cleanconll          # load a dataset, freeze its splits, print a report
active-gliner label --dataset cleanconll --teacher gemma-4-12b --split pool
active-gliner tune --dataset cleanconll --trials 20 --freeze
active-gliner matrix list                    # every run, by block
active-gliner matrix run --block ner_core    # resumable: finished runs are skipped
active-gliner analyse                        # statistics, figures, paper/numbers.tex
just arxiv                                   # pack the paper for arXiv
```

+ Datasets - NER: CleanCoNLL, BC5CDR, MIT Movie (corrected), UNER EWT. Relations: CrossRE.
  Multi-label: Hallmarks of Cancer. Slots: MASSIVE 1.1 (English and French). Decisions: BoolQ,
  CLINC150. Frozen split ids are in `data/splits/`; sources and versions are in
  `docs/research/2026-09-25-1230-dataset-sources.md`.
+ Student - `fastino/gliner2.5-multi-v1`, pinned by revision.
+ Teachers - Gemma 4 12B and Gemma 4 E4B (local, llama.cpp), and GPT-6 Luna (OpenAI API).
  Launch scripts are in `scripts/`, pins in `configs/teachers/`.
+ Design - every decision is in `docs/research/2026-09-25-1113-full-merge-design.md`.

## Development

```bash
just check            # ruff + tests
just run-smoke        # a real 20-step training run on the GPU
just notebook         # rebuild and run the quickstart notebook (GPU and teacher needed)
just task-notebooks   # rebuild and run the five task notebooks (GPU and teacher needed)
```

The notebooks come from `notebooks/build_quickstart.py` and `notebooks/build_task_notebooks.py`;
edit the builders, not the `.ipynb` files. Rules for contributors are in `AGENTS.md`.

## The thesis

This project grew out of the author's master thesis on MIT Movie. The thesis code, results and
data are at the git tag [`thesis-v1`](https://github.com/dcrey7/active_gliner/tree/thesis-v1).
The full report is
[here](https://drive.google.com/file/d/1eo1z6MbX-gSsD8jMPwdCOveldRVPxqrF/view?usp=drive_link).
What the thesis did, and what the new study fixes, is in Section 5 of the paper.

## Citation

If you use Active GLiNER, its code, its results or its ideas, in a paper, a product or a
service, cite it:

```bibtex
@software{thomas2026activegliner,
  author  = {Thomas, Abhishek},
  title   = {Active {GLiNER}: Which Examples Should {LLM}s Label?},
  year    = {2026},
  url     = {https://github.com/dcrey7/active_gliner},
  license = {Apache-2.0}
}
```

GitHub shows the same entry under "Cite this repository" (from `CITATION.cff`).

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). See
[NOTICE](NOTICE).

Under section 4 of the licence, anyone who redistributes this code or a work derived from it
must keep the copyright notice and include the attribution in the [NOTICE](NOTICE) file.

Datasets, model weights (GLiNER, GLiNER2, and others), and LLM outputs are not covered by this
licence. Each keeps the licence and terms of its owner. Check them before you reuse or
redistribute data or labels. CleanCoNLL and MIT Movie carry no open licence: we release their
labels only as sentence ids and offsets.
