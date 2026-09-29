import json
from copy import deepcopy
from dataclasses import replace

from gliner2.processing.word_splitter import WhitespaceTokenSplitter

from active_gliner.data.records import Record


def empty_gold(task: str) -> dict:
    if task == "relations":
        return {"entities": [], "relations": []}
    if task == "classification":
        return {"labels": []}
    if task in {"ner", "slots"}:
        return {"spans": []}
    raise ValueError(f"Unknown task: {task}")


def span_decision(candidate: dict, accepted: list[dict]) -> str:
    """Keep the first listed span; distinguish repetition from conflicting overlap."""
    if candidate in accepted:
        return "duplicate"
    if any(candidate["start"] < old["end"] and old["start"] < candidate["end"] for old in accepted):
        return "overlap_conflict"
    return "accept"


def parse(
    record: Record, task: str, content: str, labels: dict[str, str], counts: dict | None = None
) -> tuple:
    gold = empty_gold(task)
    if task == "relations":
        gold["entities"] = deepcopy(record.gold["entities"])
    errors = []

    def error(kind, item):
        errors.append({"kind": kind, "item": item})

    try:
        # The outer braces preserve braces inside JSON strings, unlike brace counting.
        start, end = content.find("{"), content.rfind("}")
        value = json.loads(content[start : end + 1] if start >= 0 else content)
    except (ValueError, AttributeError, TypeError) as exc:
        return None, [{"kind": "json", "message": str(exc)}]
    key = {
        "ner": "entities",
        "relations": "relations",
        "classification": "labels",
        "slots": "slots",
    }[task]
    if value == []:
        return gold, []
    expected = dict if task == "slots" else list
    if not isinstance(value, dict) or not isinstance(value.get(key), expected):
        return None, [{"kind": "schema", "message": f"Expected {key}: {expected.__name__}"}]
    raw = {natural: name for name, natural in labels.items()}
    words = list(WhitespaceTokenSplitter()(record.text))
    word_starts = {start for _, start, _ in words}
    word_ends = {end for _, _, end in words}

    def allowed(label):
        if not isinstance(label, str):
            error("schema", label)
            return False
        if label not in raw:
            error("label_not_allowed", label)
            return False
        return True

    def occurrences(text):
        if not isinstance(text, str) or not text:
            error("schema", text)
            return []
        found, spans, offset = False, [], 0
        while (start := record.text.find(text, offset)) >= 0:
            offset = start + 1
            found = True
            end = start + len(text)
            # A match inside a word ("r" in "rated") is not the mention the teacher named,
            # and the student cannot learn a span that cuts one of its words.
            if start in word_starts and end in word_ends:
                spans.append({"start": start, "end": end, "text": text})
        if not found:
            error("not_in_text", text)
        elif not spans:
            error("not_on_word_edges", text)
        return spans

    def mention(text, label):
        if not allowed(label):
            return
        for span in occurrences(text):
            candidate = {**span, "label": raw[label]}
            decision = span_decision(candidate, gold["spans"])
            if counts is not None:
                counts[decision] = counts.get(decision, 0) + 1
            if decision == "overlap_conflict":
                error(decision, candidate)
            elif decision == "accept":
                gold["spans"].append(candidate)

    items = value[key]
    if task == "classification":
        accepted = {raw[label] for label in items if allowed(label)}
        gold["labels"] = [label for label in labels if label in accepted]
    elif task == "slots":
        for label, texts in items.items():
            if not allowed(label):
                continue
            if not isinstance(texts, list):
                error("schema", texts)
                continue
            for text in texts:
                mention(text, label)
    elif task == "relations":
        entities = gold["entities"]
        for item in items:
            if not isinstance(item, dict):
                error("schema", item)
                continue
            head, tail = item.get("head"), item.get("tail")
            if (
                any(type(i) is not int or not 0 <= i < len(entities) for i in (head, tail))
                or head == tail
            ):
                error("bad_mention", item)
                continue
            if not isinstance(item.get("types"), list):
                error("schema", item)
                continue
            for label in item["types"]:
                if not allowed(label):
                    continue
                relation = {
                    "head_id": head,
                    "tail_id": tail,
                    "type": raw[label],
                    "head": {k: entities[head][k] for k in ("start", "end", "text")},
                    "tail": {k: entities[tail][k] for k in ("start", "end", "text")},
                }
                if relation not in gold["relations"]:
                    gold["relations"].append(relation)
    else:
        for item in items:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(f), str) for f in ("text", "label")
            ):
                error("schema", item)
                continue
            mention(item["text"], item["label"])
    return gold, errors


def teacher_records(records: list[Record], labelled: dict) -> list[Record]:
    """Keep every selected record, including invalid responses counted by TeacherStats."""
    return [
        replace(
            record,
            gold=deepcopy(labelled[record.id]["gold"])
            if labelled[record.id]["gold"] is not None
            else (
                {"entities": deepcopy(record.gold["entities"]), "relations": []}
                if record.task == "relations"
                else empty_gold(record.task)
            ),
        )
        for record in records
    ]
