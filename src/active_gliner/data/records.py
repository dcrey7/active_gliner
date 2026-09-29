from dataclasses import dataclass


@dataclass
class Record:
    id: str
    doc_id: str | None
    text: str
    task: str
    locale: str
    source_split: str
    gold: dict


def gold_spans(record: Record) -> list[dict]:
    if record.task == "relations":
        return record.gold["entities"] + [
            relation[side] for relation in record.gold["relations"] for side in ("head", "tail")
        ]
    return record.gold.get("spans", [])
