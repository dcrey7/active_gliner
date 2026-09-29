"""Write notebooks/quickstart.ipynb from the cells below.

Edit the cells here, then run: uv run python notebooks/build_quickstart.py
Run the notebook with: just notebook
"""

import json
import sys
from pathlib import Path

CELLS = [
    (
        "markdown",
        """# Active GLiNER quickstart

Train a small, fast extraction model on labels from an LLM.

```
your texts ──> the student scores them ──> pick N texts ──> the LLM labels them
                                                                  │
   a fast model you own <── LoRA training on those N labels <─────┘
```

This notebook uses MIT Movie questions (for example *"show me films with drew barrymore
from the 1980s"*) and 12 entity types. It shows:

1. What the untrained student finds (zero-shot).
2. How to plan a run without a GPU.
3. How to train on 200 LLM labels with `fit`.
4. How to read the report and use the trained model with `predict`.

You need a GPU (8 GB is enough) and an OpenAI-compatible LLM server. This repository
starts Gemma 4 12B with `scripts/serve_teacher_gemma.sh`; any other local or hosted
endpoint works too.""",
    ),
    (
        "code",
        """import json
from pathlib import Path

import torch
from IPython.display import Image, Markdown, display

from active_gliner import byod, data, model, tasks

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")""",
    ),
    (
        "markdown",
        """## 1. Your data

`fit` reads plain texts: one per line, or JSONL with a `text` field. A small labelled dev
file is optional but recommended: without it the report shows agreement with the LLM,
not accuracy.

Here we write both files from MIT Movie: 1,000 unlabelled questions and 200 labelled ones.""",
    ),
    (
        "code",
        """demo = Path("demo")
demo.mkdir(exist_ok=True)
splits = data.load("mit_movie")
names = tasks.labels("mit_movie", splits)  # {"ACTOR": "actor", ...}

with open(demo / "texts.jsonl", "w") as f:
    for record in splits["pool"][:1000]:
        f.write(json.dumps({"text": record.text}) + "\\n")

with open(demo / "dev.jsonl", "w") as f:
    for record in splits["dev"][:200]:
        entities = [{**span, "label": names[span["label"]]} for span in record.gold["spans"]]
        f.write(json.dumps({"text": record.text, "entities": entities}) + "\\n")

# Label names, each with an optional short definition after ":".
labels = ",".join(names.values())
print(labels)
print(open(demo / "dev.jsonl").readline())""",
    ),
    (
        "markdown",
        """## 2. What the untrained student finds

GLiNER2.5 already extracts entities for any label names, with no training.""",
    ),
    (
        "code",
        """ner = tasks.for_dataset("mit_movie")
student = model.load_student()
dev = byod.read_labelled(demo / "dev.jsonl", "ner")
label_map = byod.parse_labels(labels)
zero_shot = ner.predict(student, dev, label_map)

for record, prediction in list(zip(dev, zero_shot))[:3]:
    print(record.text)
    print("   found:", [(s["text"], s["label"]) for s in prediction["spans"]])

zero_shot_f1 = ner.score(dev, zero_shot, label_map)["micro"]["f1"]
print(f"\\nZero-shot dev F1: {100 * zero_shot_f1:.1f}")
del student
torch.cuda.empty_cache()""",
    ),
    (
        "markdown",
        """## 3. Plan the run

`plan` checks the files and settings without loading a model.

+ `n` - how many texts the LLM labels.
+ `select` - `random`, or `min` to pick the texts the student is least sure about.
  In the paper, random was as good or better for NER, so we use it here.""",
    ),
    (
        "code",
        """byod.plan(demo / "texts.jsonl", "ner", labels, n=200, select="random", seed=1)""",
    ),
    (
        "markdown",
        """## 4. Label and train

`fit` scores the texts, picks 200, asks the LLM to label them, checks every answer
(valid JSON, allowed labels, exact text spans), and trains a LoRA adapter. Change
`TEACHER_URL` and `TEACHER_MODEL` for your own server. For a hosted API, also pass
`api_key_env="NAME_OF_THE_KEY_VARIABLE"`.

The same run from a terminal:

```bash
active-gliner fit --texts demo/texts.jsonl --task ner --labels "actor,director,..." \\
  --teacher-url http://127.0.0.1:8020/v1 --teacher-model gemma-4-12b-qat \\
  --n 200 --select random --dev demo/dev.jsonl --out demo/run
```""",
    ),
    (
        "code",
        """TEACHER_URL = "http://127.0.0.1:8020/v1"
TEACHER_MODEL = "gemma-4-12b-qat"

run = byod.fit(
    texts=demo / "texts.jsonl",
    task="ner",
    labels=labels,
    teacher_url=TEACHER_URL,
    teacher_model=TEACHER_MODEL,
    n=200,
    select="random",
    dev=demo / "dev.jsonl",
    max_steps=300,
    eval_steps=50,
    out_dir=demo / "run",
)
print(run)""",
    ),
    (
        "markdown",
        """## 5. Read the results

Every run folder holds plain files: the report, training curves, every LLM label and
every wrong prediction.""",
    ),
    (
        "code",
        """metrics = json.loads((run / "metrics.json").read_text())
trained_f1 = metrics["dev"]["micro"]["f1"]
print(f"Dev F1: zero-shot {100 * zero_shot_f1:.1f} -> trained {100 * trained_f1:.1f}")
print("LLM answers that passed the checks:", json.loads((run / "teacher_stats.json").read_text()))
print(sorted(p.name for p in run.iterdir()))""",
    ),
    ("code", """display(Markdown((run / "report.md").read_text()))"""),
    ("code", """display(Image(filename=str(run / "plots" / "training_curves.png")))"""),
    (
        "markdown",
        """## 6. Use the trained model

`predict` loads the adapter and runs on a GPU or a CPU. From a terminal:

```bash
active-gliner predict --adapter demo/run/adapter/best --task ner \\
  --labels "actor,director,..." --texts new.jsonl
```""",
    ),
    (
        "code",
        """questions = [
    "who directed the dark knight",
    "any good horror movies from 1978 with jamie lee curtis",
    "play the trailer for the new pixar film",
]
(demo / "new.jsonl").write_text("".join(json.dumps({"text": q}) + "\\n" for q in questions))

for question, prediction in zip(
    questions, byod.predict(run / "adapter" / "best", "ner", labels, demo / "new.jsonl")
):
    print(question)
    print("   ", [(s["text"], s["label"]) for s in prediction["spans"]])""",
    ),
    (
        "markdown",
        """## Other tasks

The same two calls work for the other tasks with `task=`:

| Task | Input file | Labels |
|---|---|---|
| `classification` | texts | the class names; a text can get several or none |
| `relations` | texts with an `entities` list (the mentions) | the relation types |
| `slots` | texts (commands) | the slot names; the output is JSON |

The paper's experiments use the same code through `active-gliner matrix run`; see the README.""",
    ),
]


def cell(index: int, kind: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "id": f"cell-{index}", "metadata": {}, "source": lines}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cell-{index}",
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


path = Path(__file__).with_name("quickstart.ipynb")
if sys.argv[1:] == ["--drop-stderr"]:
    # After a run: warnings and progress bars carry local paths and add nothing to read.
    executed = json.loads(path.read_text(encoding="utf-8"))
    for index, item in enumerate(executed["cells"]):
        item.setdefault("id", f"cell-{index}")
        if item["cell_type"] == "code":
            item["outputs"] = [o for o in item["outputs"] if o.get("name") != "stderr"]
        else:
            item.pop("outputs", None)
    path.write_text(json.dumps(executed, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.exit(0)

notebook = {
    "cells": [cell(index, kind, text) for index, (kind, text) in enumerate(CELLS)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(path)
