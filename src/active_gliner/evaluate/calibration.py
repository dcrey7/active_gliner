"""Four reliability bins using emitted spans or all classification and pair-label decisions."""

from .errors import items


def calibration_rows(task, records, predictions) -> list[dict]:
    bins = [[] for _ in range(4)]
    for record, prediction in zip(records, predictions, strict=True):
        gold = task.gold_items(record)
        for key, _item, confidence in items(task, prediction):
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid prediction confidence: {confidence}")
            bins[min(int(confidence * 4), 3)].append((confidence, key in gold))
    rows = []
    for index, values in enumerate(bins):
        count = len(values)
        rows.append(
            {
                "bin": f"[{index / 4:.2f},{(index + 1) / 4:.2f}" + ("]" if index == 3 else ")"),
                "count": count,
                "mean_confidence": sum(p for p, _ in values) / count if count else None,
                "correctness": sum(correct for _, correct in values) / count if count else None,
            }
        )
    return rows
