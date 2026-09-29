"""Exact-match metrics for structured predictions."""


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _report(counts: dict) -> dict:
    per_label = {
        label: {**_prf(tp, fp, fn), "support": tp + fn} for label, (tp, fp, fn) in counts.items()
    }
    totals = [sum(row[i] for row in counts.values()) for i in range(3)]
    return {
        "micro": _prf(*totals),
        "macro_f1": (
            sum(row["f1"] for row in per_label.values()) / len(per_label) if per_label else 0.0
        ),
        "per_label": per_label,
    }


def span_prf(gold: list[list[tuple]], pred: list[list[tuple]]) -> dict:
    """Count exact matches per record, with the label last in each tuple."""
    counts = {}
    for expected, actual in zip(gold, pred, strict=True):
        expected, actual = set(expected), set(actual)
        for index, items in enumerate((expected & actual, actual - expected, expected - actual)):
            for item in items:
                counts.setdefault(item[-1], [0, 0, 0])[index] += 1
    return _report(counts)


def classification_prf(gold: list[set], pred: list[set], labels: list[str]) -> dict:
    """Count positive decisions over the given labels, including absent labels in macro F1."""
    counts = {label: [0, 0, 0] for label in labels}
    for expected, actual in zip(gold, pred, strict=True):
        for label, row in counts.items():
            row[0] += int(label in expected and label in actual)
            row[1] += int(label not in expected and label in actual)
            row[2] += int(label in expected and label not in actual)
    return _report(counts)


def canonical_value(s: str) -> str:
    """Casefold slot values and collapse whitespace."""
    return " ".join(s.casefold().split())


def record_accuracy(gold: list[set], pred: list[set]) -> float:
    """Return the fraction of records with exactly matching sets."""
    matches = sum(expected == actual for expected, actual in zip(gold, pred, strict=True))
    return matches / len(gold) if gold else 0.0
