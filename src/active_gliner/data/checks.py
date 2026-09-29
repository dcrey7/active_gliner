from collections import Counter

from . import load
from .records import gold_spans
from .splits import split_hash


def check(name: str, locale: str = "en-US") -> dict:
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    splits = load(name, locale)
    report = {
        "dataset": name,
        "locale": locale,
        "sizes": {},
        "label_counts": {},
        "empty_gold": {},
        "split_hash": split_hash(splits),
    }
    misaligned = total = arguments = repeated = relation_free = records_count = 0
    splitter = WhitespaceTokenSplitter()
    for split, records in splits.items():
        counts = Counter()
        empty = 0
        for record in records:
            if record.task == "relations":
                labels = [r["type"] for r in record.gold["relations"]]
                relation_free += not labels
                records_count += 1
                for relation in record.gold["relations"]:
                    for side in ("head", "tail"):
                        value = relation[side]["text"]
                        first = record.text.find(value)
                        repeated += record.text.find(value, first + 1) >= 0
                        arguments += 1
            elif record.task == "classification":
                labels = record.gold["labels"]
            else:
                labels = [span["label"] for span in record.gold["spans"]]
            counts.update(labels)
            empty += not labels
            spans = gold_spans(record)
            for span in spans:
                start, end = span["start"], span["end"]
                if (
                    not 0 <= start < end <= len(record.text)
                    or record.text[start:end] != span["text"]
                ):
                    raise ValueError(f"Invalid gold span: {record.id}: {span}")
            if split == "dev":
                tokens = list(splitter(record.text))
                starts, ends = {t[1] for t in tokens}, {t[2] for t in tokens}
                total += len(spans)
                misaligned += sum(s["start"] not in starts or s["end"] not in ends for s in spans)
        report["sizes"][split] = len(records)
        report["label_counts"][split] = dict(sorted(counts.items()))
        report["empty_gold"][split] = empty
    report["misaligned_dev_rate"] = misaligned / total if total else 0.0
    if name == "crossre":
        report["repeated_argument_rate"] = repeated / arguments if arguments else 0.0
        report["relation_free_rate"] = relation_free / records_count if records_count else 0.0
    return report


def altered_target_rate(dataset, locale: str = "en-US") -> dict:
    """Measure old surface matching: gold spans with at least one non-gold match."""
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    splits = load(dataset, locale) if isinstance(dataset, str) else dataset
    if not isinstance(splits, dict):
        splits = {"records": splits}
    splitter = WhitespaceTokenSplitter()
    report = {}
    for split, records in splits.items():
        altered = total = 0
        for record in records:
            tokens = list(splitter(record.text))
            words = [t[0] for t in tokens]
            gold = {(s["start"], s["end"], s["label"]) for s in record.gold["spans"]}
            for span in record.gold["spans"]:
                surface = [t[0] for t in splitter(span["text"])]
                size = len(surface)
                total += 1
                altered += any(
                    words[i:i + size] == surface
                    and (tokens[i][1], tokens[i + size - 1][2], span["label"]) not in gold
                    for i in range(len(words) - size + 1)
                ) if size else False
        report[split] = {"altered": altered, "total": total,
                         "rate": altered / total if total else 0.0}
    altered = sum(row["altered"] for row in report.values())
    total = sum(row["total"] for row in report.values())
    return {"splits": report, "altered": altered, "total": total,
            "rate": altered / total if total else 0.0}
