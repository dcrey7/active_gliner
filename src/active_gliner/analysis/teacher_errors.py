"""What kind of mistakes does each teacher make on test? Descriptive, read from the label cache.

Span tasks: each ground-truth span is exact, wrong edges, wrong type, wrong edges and type,
or missed; a teacher span that overlaps no ground-truth span is invented. Hallmarks: label
precision and recall, and how often a "none" sentence gets a label. These explain why some
tasks tolerate LLM labels and others do not (labels that follow corpus conventions).
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from active_gliner import data, tasks

SPAN_PLACES = {
    ("cleanconll", "en-US"): ("CleanCoNLL", "CleanCoNLL"),
    ("bc5cdr", "en-US"): ("BC5CDR", "BCfiveCDR"),
    ("mit_movie", "en-US"): ("MIT Movie", "MITMovie"),
    ("uner_ewt", "en-US"): ("UNER EWT", "UnerEwt"),
    ("massive", "en-US"): ("MASSIVE en", "MassiveEn"),
    ("massive", "fr-FR"): ("MASSIVE fr", "MassiveFr"),
}
TEACHERS = {"gemma-4-12b": "Gemma", "gpt-6-luna": "Luna"}


def teacher_labels(teacher: str, dataset: str, locale: str, root="labels") -> dict:
    """The teacher's validated labels from the cache of the frozen prompt."""
    from active_gliner.teachers import prompts

    splits = data.load(dataset, locale)
    labels = tasks.labels(dataset, splits)
    task = tasks.for_dataset(dataset).__name__.rsplit(".", 1)[-1]
    # The same rule as teachers.client: prompt v1 sends no label definitions.
    definitions = None if prompts.version_for(dataset) == "v1" else prompts.definitions_for(dataset)
    digest = prompts.prompt_hash(task, labels, definitions)
    path = Path(root) / teacher / f"{dataset}-{locale}" / f"{task}-{digest}.jsonl"
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            out[row["id"]] = row["value"]["gold"]
    return out


def _overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def span_errors(records, labels: dict) -> dict:
    kinds = Counter()
    alarms = empty = 0
    for record in records:
        if labels.get(record.id) is None:
            continue
        gold = record.gold["spans"]
        pred = labels[record.id].get("spans", [])
        if not gold:
            empty += 1
            alarms += bool(pred)
        for g in gold:
            match = next((p for p in pred if _overlap(g, p)), None)
            if match is None:
                kinds["missed"] += 1
                continue
            same_edges = (match["start"], match["end"]) == (g["start"], g["end"])
            same_type = match["label"] == g["label"]
            kinds[
                "exact"
                if same_edges and same_type
                else "type"
                if same_edges
                else "edges"
                if same_type
                else "edges_type"
            ] += 1
        kinds["invented"] += sum(1 for p in pred if not any(_overlap(p, g) for g in gold))
    total = sum(kinds[k] for k in ("exact", "type", "edges", "edges_type", "missed"))
    share = {
        k: 100 * kinds[k] / total
        for k in ("exact", "type", "edges", "edges_type", "missed", "invented")
    }
    return {**share, "empty_alarm": 100 * alarms / empty if empty else 0.0, "gold": total}


def label_errors(records, labels: dict) -> dict:
    tp = fp = fn = alarms = empty = 0
    for record in records:
        if labels.get(record.id) is None:
            continue
        gold, pred = set(record.gold["labels"]), set(labels[record.id].get("labels", []))
        tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
        if not gold:
            empty, alarms = empty + 1, alarms + bool(pred)
    return {
        "precision": 100 * tp / (tp + fp),
        "recall": 100 * tp / (tp + fn),
        "empty_alarm": 100 * alarms / empty,
    }


def rows(root="labels") -> pd.DataFrame:
    out = []
    for teacher, tag in TEACHERS.items():
        for (dataset, locale), (title, name) in SPAN_PLACES.items():
            test = data.load(dataset, locale)["test"]
            found = span_errors(test, teacher_labels(teacher, dataset, locale, root))
            out.append(dict(title=title, name=name, teacher=tag, **found))
        test = data.load("hallmarks", "en-US")["test"]
        found = label_errors(test, teacher_labels(teacher, "hallmarks", "en-US", root))
        out.append(dict(title="Hallmarks", name="Hallmarks", teacher=tag, **found))
    return pd.DataFrame(out)


def macros(table: pd.DataFrame) -> dict:
    keys = {
        "exact": "Exact",
        "type": "WrongType",
        "edges": "WrongEdges",
        "edges_type": "WrongBoth",
        "missed": "Missed",
        "invented": "Invented",
        "empty_alarm": "EmptyAlarm",
        "precision": "Precision",
        "recall": "Recall",
    }
    found = {}
    for row in table.to_dict("records"):
        for key, suffix in keys.items():
            if key in row and not pd.isna(row[key]):
                found[row["name"] + row["teacher"] + "TeacherErr" + suffix] = round(row[key])
    return found


def latex_table(table: pd.DataFrame, path) -> Path:
    """Span tasks only: shares of ground-truth spans, invented spans and empty alarms."""
    lines = [
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Dataset & Teacher & Exact & Edges & Type & Both & Missed & Invented & Empty \\\\",
        "\\midrule",
    ]
    for row in table[table.title != "Hallmarks"].to_dict("records"):
        cells = [
            row[k]
            for k in ("exact", "edges", "type", "edges_type", "missed", "invented", "empty_alarm")
        ]
        lines.append(
            f"{row['title']} & {row['teacher']} & "
            + " & ".join(f"{v:.0f}" for v in cells)
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def org_as_loc_macros(root="labels") -> dict:
    """CleanCoNLL: share of same-span type errors that are organisations labelled as places.

    CoNLL marks sports teams named after a city or country ("Karlsruhe 3 Freiburg 0") as
    organisations; a teacher without the guidelines reads them as locations.
    """
    found = {}
    for teacher, tag in TEACHERS.items():
        labels = teacher_labels(teacher, "cleanconll", "en-US", root)
        pairs = Counter()
        for record in data.load("cleanconll", "en-US")["test"]:
            pred = (labels.get(record.id) or {}).get("spans", [])
            for g in record.gold["spans"]:
                for p in pred:
                    same_edges = (p["start"], p["end"]) == (g["start"], g["end"])
                    if same_edges and p["label"] != g["label"]:
                        pairs[(g["label"], p["label"])] += 1
        total = sum(pairs.values())
        if total:
            found[f"CleanCoNLL{tag}TeacherErrOrgAsLoc"] = round(100 * pairs[("ORG", "LOC")] / total)
    return found


def majority_macros() -> dict:
    """Test accuracy of always giving the most common answer (BoolQ: "yes")."""
    test = data.load("boolq", "en-US")["test"]
    counts = Counter(record.gold["labels"][0] for record in test)
    return {"BoolQMajority": 100 * max(counts.values()) / len(test)}


def boolq_length_macros() -> dict:
    """The longest BoolQ input in words (question plus passage, any split); it sets GPU memory."""
    splits = data.load("boolq", "en-US")
    longest = max(len(record.text.split()) for split in splits.values() for record in split)
    return {"BoolQMaxInputWords": f"{longest:,}"}
