"""Write one notebook per task type, each following the paper's recipe.

Edit the cells here, then run: uv run python notebooks/build_task_notebooks.py
Run the notebooks with: just task-notebooks (GPU and teacher needed)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

TASKS = {
    "ner": {
        "dataset": "mit_movie",
        "title": "Named entities (NER)",
        "what": "find names and other short spans in a text, each with a type",
        "example": '*"show me films with drew barrymore from the 1980s"* gives '
        "`drew barrymore` = actor and `1980s` = year",
        "data": "MIT Movie: questions about films, 12 entity types",
        "pool": 5000,
        "new": [
            "who directed the dark knight",
            "any good horror movies from 1978 with jamie lee curtis",
            "play the trailer for the new pixar film",
        ],
        "show": 'print("   ", [(s["text"], s["label"]) for s in prediction["spans"]])',
        "paper": "On our four NER datasets, LLM labels for a quarter of the inputs cost under two "
        "F1 points, and a quarter or fewer human labels beat the LLM itself on most datasets.",
    },
    "relations": {
        "dataset": "crossre",
        "title": "Relations between given mentions",
        "what": "say how two mentions in a sentence are related; the mentions are given",
        "example": "in a sentence that names *Marie Curie* and *Paris*, the pair can get the "
        "relation `physical` (she lived there)",
        "data": "CrossRE: sentences from six domains, 17 relation types; every sentence comes "
        "with its entity mentions",
        "pool": 2519,
        "new": None,
        "show": 'mention = [e["text"] for e in row["entities"]]\n    '
        'print("   ", [(mention[r["head"]], r["type"], mention[r["tail"]]) '
        'for r in prediction["relations"]])',
        "paper": "Relations are hard for both the LLM and the student (F1 below 50). "
        "LLM labels cost more here than on NER: 2 to 3 points for a quarter of LLM labels.",
    },
    "classification": {
        "dataset": "hallmarks",
        "title": "Multi-label classification",
        "what": "give a text all the labels that apply, or none",
        "example": "a sentence about tumour cells that escape the immune system gets "
        "`avoiding immune destruction`",
        "data": "Hallmarks of Cancer: sentences from cancer papers, 10 hallmarks; many "
        "sentences have none",
        "pool": 5000,
        "new": [
            "Tumour cells evade T cell attack by expressing PD-L1 .",
            "The mutation increases glucose uptake and shifts metabolism to glycolysis .",
            "Samples were stored at minus 80 degrees until use .",
        ],
        "show": 'print("   ", prediction["labels"])',
        "paper": "The LLM often gives a hallmark to sentences the expert marked as none, so this "
        "task needs more human labels: about three quarters of 1,000 ranked sentences.",
    },
    "slots": {
        "dataset": "massive",
        "title": "Slot filling to JSON",
        "what": "turn a voice command into JSON slots",
        "example": '*"wake me up at 7 am tomorrow"* gives '
        '`{"time": ["7 am"], "date": ["tomorrow"]}`',
        "data": "MASSIVE 1.1 English: voice-assistant commands, 55 slot types",
        "pool": 5000,
        "new": [
            "wake me up at seven tomorrow morning",
            "play some jazz by miles davis",
            "what is the weather like in paris this weekend",
        ],
        "show": 'print("   ", [(s["text"], s["label"]) for s in prediction["spans"]])',
        "paper": "Slots tolerate LLM labels the least: the LLM fills slots in most commands that "
        "have none, and a quarter of LLM labels already costs about 6 F1 points. Use a larger "
        "human share here.",
    },
    "decision": {
        "dataset": "clinc150",
        "title": "Decisions: one answer from a list",
        "what": "pick exactly one answer from a fixed list of options",
        "example": '*"how much is left on my visa card"* gives the intent `balance`',
        "data": "CLINC150: assistant requests, 150 intents plus `oos` (out of scope)",
        "pool": 5000,
        "new": [
            "how much money is in my checking account",
            "set a timer for ten minutes",
            "what is the meaning of life",
        ],
        "show": 'print("   ", prediction["labels"])',
        "paper": "Decisions tolerate LLM labels best: half LLM labels cost under one point. But on "
        "CLINC150 the LLM itself stays ahead of the small student, mostly on out-of-scope "
        "requests, so check that the student is good enough for your use.",
    },
}

CELLS = [
    (
        "markdown",
        """# Active GLiNER: __TITLE__

The task: __WHAT__. For example, __EXAMPLE__.

This notebook follows the recipe from the paper:

```
your unlabelled texts
   │  1. the untrained student ranks them once, least sure first
   ▼
top N texts ──> a random quarter: human labels
           └──> the rest: LLM labels
   │  2. train the student on both (LoRA)
   ▼
a small, fast model you own ──> 3. check it on labelled dev data ──> 4. predict
```

Data: __DATA__. The "human" labels here come from the dataset's own annotators, so you
can see the recipe end to end without labelling anything yourself.

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

from active_gliner import byod, data, tasks

TASK = "__TASK__"
DATASET = "__DATASET__"
TEACHER_URL = "http://127.0.0.1:8020/v1"
TEACHER_MODEL = "gemma-4-12b-qat"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")""",
    ),
    (
        "markdown",
        """## 1. Your data

The tool reads JSONL files, one text per line:

+ `texts.jsonl` - unlabelled texts (`text`, and for relations the `entities` list).
+ `dev.jsonl` - a few hundred texts with correct labels, to measure accuracy.

Here we write both from the dataset.""",
    ),
    (
        "code",
        """demo = Path("demo-__TASK__")
demo.mkdir(exist_ok=True)
splits = data.load(DATASET)
names = tasks.labels(DATASET, splits)  # label code -> label name


def write(path, records, with_labels):
    rows = [byod.record_to_row(r, names, with_labels) for r in records]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows))


write(demo / "texts.jsonl", splits["pool"][:__POOL__], with_labels=False)
write(demo / "dev.jsonl", splits["dev"][:300], with_labels=True)
labels = ",".join(names.values())
print(len(names), "labels:", labels[:200], "...")
print(open(demo / "dev.jsonl").readline()[:300])""",
    ),
    (
        "markdown",
        """## 2. What the untrained student does

GLiNER2.5 already works for any label names, with no training. This is the score to beat.""",
    ),
    (
        "code",
        """task = tasks.for_dataset(DATASET)
dev = byod.read_labelled(demo / "dev.jsonl", TASK)
label_map = byod.parse_labels(labels)
student = byod.load_student(TASK, label_map)
zero_shot_f1 = task.score(dev, task.predict(student, dev, label_map), label_map)["micro"]["f1"]
print(f"Zero-shot dev F1: {100 * zero_shot_f1:.1f}")
del student
torch.cuda.empty_cache()""",
    ),
    (
        "markdown",
        """## 3. Rank once and split between humans and the LLM

`rank` scores every text with the untrained student, takes the 1,000 it is least sure
about (texts where it finds nothing go last), and puts a random quarter of them in
`for_humans.jsonl`. The rest goes in `for_llm.jsonl`.

From a terminal:

```bash
active-gliner rank --texts demo/texts.jsonl --task __TASK__ --labels "..." \\
  --n 1000 --human-share 0.25 --out demo/rank
```""",
    ),
    (
        "code",
        """ranked = byod.rank(
    demo / "texts.jsonl", TASK, labels, n=1000, human_share=0.25, out_dir=demo / "rank"
)
for_humans = [json.loads(line) for line in (ranked / "for_humans.jsonl").open()]
for_llm = [json.loads(line) for line in (ranked / "for_llm.jsonl").open()]
print(len(for_humans), "texts for humans,", len(for_llm), "for the LLM")
print("example for a human:", for_humans[0]["text"][:200])""",
    ),
    (
        "markdown",
        """## 4. Human labels

In real use, people label `for_humans.jsonl` by hand, in the same format as `dev.jsonl`.
Here we copy the dataset's own labels for those texts.""",
    ),
    (
        "code",
        """by_id = {r.id: r for r in splits["pool"]}
write(demo / "human.jsonl", [by_id[row["id"]] for row in for_humans], with_labels=True)
print(open(demo / "human.jsonl").readline()[:300])""",
    ),
    (
        "markdown",
        """## 5. LLM labels for the rest, then train

`fit` sends every text in `for_llm.jsonl` to the LLM (`select="all"`), checks every
answer (valid JSON, allowed labels, exact text spans), adds the human-labelled texts,
and trains a LoRA adapter. The LLM never sees the human-labelled texts.

```bash
active-gliner fit --texts demo/rank/for_llm.jsonl --select all --human demo/human.jsonl \\
  --task __TASK__ --labels "..." --dev demo/dev.jsonl \\
  --teacher-url http://127.0.0.1:8020/v1 --teacher-model gemma-4-12b-qat --out demo/run
```

For a hosted API, also pass `api_key_env="NAME_OF_THE_KEY_VARIABLE"`.""",
    ),
    (
        "code",
        """run = byod.fit(
    texts=ranked / "for_llm.jsonl",
    task=TASK,
    labels=labels,
    teacher_url=TEACHER_URL,
    teacher_model=TEACHER_MODEL,
    select="all",
    human=demo / "human.jsonl",
    dev=demo / "dev.jsonl",
__FIT_EXTRA__    out_dir=demo / "run",
)
print(run)""",
    ),
    (
        "markdown",
        """## 6. Check the result

Every run folder holds plain files: a one-page report, training curves, every LLM label
and every wrong prediction.""",
    ),
    (
        "code",
        """metrics = json.loads((run / "metrics.json").read_text())
trained_f1 = metrics["dev"]["micro"]["f1"]
print(f"Dev F1: zero-shot {100 * zero_shot_f1:.1f} -> trained {100 * trained_f1:.1f}")
print("LLM answers:", json.loads((run / "teacher_stats.json").read_text()))""",
    ),
    ("code", """display(Markdown((run / "report.md").read_text()))"""),
    ("code", """display(Image(filename=str(run / "plots" / "training_curves.png")))"""),
    (
        "markdown",
        """## 7. Use the trained model

`predict` loads the adapter and runs on a GPU or a CPU:

```bash
active-gliner predict --adapter demo/run/adapter/best --task __TASK__ --labels "..." \\
  --texts new.jsonl
```""",
    ),
    ("code", "__PREDICT__"),
    (
        "markdown",
        """## What the paper found for this task

__PAPER__

See the paper (`paper/main.pdf`, Section 4.4) for the numbers on every dataset.""",
    ),
]

PREDICT_NEW = """new = __NEW__
(demo / "new.jsonl").write_text("".join(json.dumps({"text": t}) + "\\n" for t in new))
predictions = byod.predict(run / "adapter" / "best", TASK, labels, demo / "new.jsonl")
for text, prediction in zip(new, predictions, strict=True):
    print(text)
    __SHOW__"""

# Relations need the mentions as input, so the examples come from the dev file.
PREDICT_DEV = """rows = [json.loads(line) for line in open(demo / "dev.jsonl")][:3]
inputs = [{"text": r["text"], "entities": r["entities"]} for r in rows]
(demo / "new.jsonl").write_text("".join(json.dumps(r) + "\\n" for r in inputs))
predictions = byod.predict(run / "adapter" / "best", TASK, labels, demo / "new.jsonl")
for row, prediction in zip(rows, predictions, strict=True):
    print(row["text"][:150])
    __SHOW__"""


def python_list(items: list[str] | None) -> str:
    """One item per line, as ruff formats it."""
    if items is None:
        return ""
    return "[\n" + "".join(f"    {json.dumps(item)},\n" for item in items) + "]"


def fill(text: str, task: str, spec: dict) -> str:
    predict = PREDICT_DEV if spec["new"] is None else PREDICT_NEW
    values = {
        "__PREDICT__": predict.replace("__NEW__", python_list(spec["new"])),
        "__SHOW__": spec["show"],
        "__TASK__": task,
        "__DATASET__": spec["dataset"],
        "__TITLE__": spec["title"],
        "__WHAT__": spec["what"],
        "__EXAMPLE__": spec["example"],
        "__DATA__": spec["data"],
        "__POOL__": str(spec["pool"]),
        "__PAPER__": spec["paper"],
        "__FIT_EXTRA__": spec.get("fit_extra", ""),
    }
    for key, value in values.items():
        text = text.replace(key, value)
    return text


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


def drop_stderr(path: Path) -> None:
    """After a run: warnings and progress bars carry local paths and add nothing to read."""
    executed = json.loads(path.read_text(encoding="utf-8"))
    for index, item in enumerate(executed["cells"]):
        item.setdefault("id", f"cell-{index}")
        if item["cell_type"] == "code":
            item["outputs"] = [o for o in item["outputs"] if o.get("name") != "stderr"]
        else:
            item.pop("outputs", None)
    path.write_text(json.dumps(executed, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    # Optional task names rebuild only those notebooks, e.g. `build_task_notebooks.py decision`.
    args = [a for a in sys.argv[1:] if a != "--drop-stderr"]
    for task, spec in TASKS.items():
        if args and task not in args:
            continue
        path = HERE / f"task_{task}.ipynb"
        if "--drop-stderr" in sys.argv[1:]:
            if path.exists():
                drop_stderr(path)
            continue
        notebook = {
            "cells": [
                cell(i, kind, fill(text, task, spec)) for i, (kind, text) in enumerate(CELLS)
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
