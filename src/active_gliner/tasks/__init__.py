"""Task dispatch and reversible label names."""

from types import ModuleType

from active_gliner.data.records import Record


class TrainingExample(dict):
    """JSONL mapping that also supports TrainingDataset's validation protocol."""

    def validate(self) -> list[str]:
        from gliner2.training.data import InputExample

        return InputExample.from_dict(self).validate()


def for_dataset(name: str) -> ModuleType:
    from . import classification, ner, relations, slots

    modules = {
        "cleanconll": ner,
        "bc5cdr": ner,
        "mit_movie": ner,
        "crossre": relations,
        "hallmarks": classification,
        "massive": slots,
    }
    if name not in modules:
        raise ValueError(f"Unknown dataset: {name}")
    return modules[name]


def labels(name: str, splits: dict[str, list[Record]]) -> dict[str, str]:
    for_dataset(name)
    raw = set()
    for record in splits["pool"]:
        if name == "crossre":
            raw.update(r["type"] for r in record.gold["relations"])
        elif name == "hallmarks":
            raw.update(record.gold["labels"])
        else:
            raw.update(s["label"] for s in record.gold["spans"])
    fixed = {
        "PER": "person",
        "LOC": "location",
        "ORG": "organization",
        "MISC": "miscellaneous",
        "Chemical": "chemical",
        "Disease": "disease",
    }
    result = {
        label: label
        if name in {"crossre", "hallmarks"}
        else fixed.get(label, label.lower().replace("_", " "))
        for label in sorted(raw)
    }
    if len(set(result.values())) != len(result):
        raise ValueError("Natural labels must be unique")
    return result


def to_raw_label(labels: dict[str, str], natural: str) -> str:
    return {value: key for key, value in labels.items()}[natural]


def pred_spans(pred: dict) -> list[dict]:
    if "spans" in pred:
        return pred["spans"]
    return [
        {**pred["entities"][relation[side]], "confidence": relation["probability"]}
        for relation in pred.get("relations", [])
        for side in ("head", "tail")
    ]
